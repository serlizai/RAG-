import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import deque
from urllib import request

from fsspec.implementations import tar
# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1:移除原生OpenAI,导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类(核心复用,替换原生OpenAI调用)
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖(消息构造+异常捕获)
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具(统一使用)
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合(小写后缀,统一匹配标准)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_support_image(filename:str) -> bool:
    """
    判断文件是否为支持的图片格式
    :param filename: 文件名含后缀
    :return: 支持True否则False
    """

    ext = Path(filename).suffix.lower()  # 取文件后缀
    # return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    return ext in IMAGE_EXTENSIONS


"""
    主要目标:对md中的图片单独处理,方便模型识别图片含义
    主要动作:图片->文件服务器minio->图片网络地址  (上文100)图片(下文100)->视觉模型->图片总结
                -> [图片总结](网络图片地址)->state->md_content = 处理后的新内容  || md_path = 处理后的路径
                步骤2(识别图片) → 产出 [(图片名, 图片地址, (上文, 下文)), ...]
                                    ↓
                步骤3(视觉总结) → 用 图片地址 + (上文, 下文) → 调用VL模型生成描述
                                    ↓
                步骤4(上传MinIO)→ 用 图片地址 上传 → 得到网络URL
                            用 图片名 生成 {图片名: 总结} 字典的 key
                                    ↓
                步骤5(替换md内容)→ 用 图片名 匹配原md中的图片引用
                            用 (上文, 下文) 定位图片在md中的位置
    技术:minio 视觉模型提示词+访问
    步骤:
        1. 校验并且获取本次操作的数据
           参数:state.md_content  state.md_path
           响应:校验后的md_content md_path 获取图片的文件夹
        2. 识别md中使用过的图片
           参数:md_content images图片的文件夹地址
           返回:[(图片名,图片地址,(上文,下文)), ...] 元组元素；地址和上下文给视觉模型总结,名和总结做成字典存储
        3. 进行图片总结和处理(视觉模型)
           参数: [(图片名,图片地址,(上文,下文)), ...] || md文件名(放入提示词,存储图片的文件夹名称)
           响应: {图片名:总结, ...}  字典
        4. 上传图片到minio和更新md_content
           参数: {图片名:总结, ...} || md_content(旧) || minio_client || [(图片名,图片地址,(上文,下文)), ...] || md文件名
           响应: new_md_content state['md_content'] = new_md_content
        5. 数据处理和备份
           参数:new_md_content 原md地址->xx.md->xx_new.md
           响应:新的md地址 new_md_path
           state['md_path'] = new_md_path
           return state
"""
def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取md内容
    :param state
    :return: md_content, md_path_obj, images_dir_obj
    """
    # 1.获取md地址
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("md地址不能为空")  # value是没有值，notfound是有值但找不到文件
    
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():  # 检查磁盘上是否真的有这个文件
        raise FileNotFoundError(f"md文件不存在: {md_file_path}")
    
    # 读取md_content
    if state.get("md_content"):
        # 已存在（如从PDF转换节点赋值），直接使用
        md_content = state["md_content"]
    else:
        # 不存在则从文件读取
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_content"] = md_content
    
    # 图片文件夹地址
    # 注意:自己传的md图片文件夹也要叫images,否则会找不到图片
    images_dir_obj = md_path_obj.parent / "images"

    return md_content, md_path_obj, images_dir_obj



def find_image_in_md_content(md_content: str, image_file: str, context_length:int = 100) -> list[Tuple[str, str]]:
    """
    在md_content中获取上下文,
    :param md_content: md内容
    :param image_file: 图片文件名
    :param context_length: 上下文长度,约定100
    :return: (上文, 下文) 如果没找到返回None
    """

    # 定义正则表达式
    # r 全称是 raw string 原始字符串
    # 作用是：告诉 Python 解释器：不要处理字符串里的转义字符（如 \、\n、\t 等），按字面意思解析
    # re.escape作用是：把一个字符串里所有可能被正则引擎当作“特殊符号”的字符全部加上反斜杠转义
    # 让它变成一个完全按字面匹配的普通字符串
    # ]在前面\[未闭合的时候不需要转义
    pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(image_file) + r".*?\)")  # 注意：三个字符串片段都要加 r 前缀

    results = []  # 一个图片可能多次使用
    # 查询图片所有匹配的位置
    for item in pattern.finditer(md_content):  # 返回的是列表，因为图片可能用多次
        start, end = item.span()  # 获取匹配的起始和结束位置
        # 截取上下文，要考虑会不会越界
        context_before = md_content[max(0, start - context_length):start]  # 上文
        context_after = md_content[end:min(len(md_content), end + context_length)]  # 下文
        results.append((context_before, context_after))

    if not results:
        logger.info(f"MD内容中未找到图片[{image_file}]的引用")
    else:
        logger.debug(f"MD内容中找到图片[{image_file}]的引用，共{len(results)}处, 截取第一个")
    return results


def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描md_content中使用过的图片,并获取图片对应的上下文
    :param md_content: md内容
    :param images_dir_obj: 图片文件夹地址
    :return: [(图片名, 图片地址, (上文, 下文)), ...]
    """
    targets = []
    # 循环读取images中的所有图片，校验在md中是否用过，用了就截取上下文
    for image_file in os.listdir(images_dir_obj):
        # 遍历每个文件的名字
        # 检查图片是否可用
        if not is_support_image(image_file):
            logger.warning(f"跳过不支持的图片格式: {image_file}")
            continue
        # 支持的话就查询md中是否存在，存在就读取对应的上下文
        content_data = find_image_in_md_content(md_content, image_file)
        if not content_data:
            logger.info(f"md中未使用图片: {image_file}, 跳过")
            continue

        # content_data = (上文, 下文),取第一个
        targets.append((image_file, str(images_dir_obj / image_file), content_data[0]))

    return targets


def step_3_generate_img_summaries(targets, stem):
    """
    利用视觉模型获取图片总结
    :param targets: [(图片名, 图片地址, (上文, 下文)), ...]
    :param stem: md文件名(不带后缀) 文件夹的名称 output/h180xxx/h180xxx.md
    :return: {图片名:总结, ...}
    """

    summaries = {}  # 最终结果
    # 循环获取图片总结
    request_times = deque()  # 用于记录请求时间戳，控制访问频率
    for image_file, image_path, content_data in targets:
        # 解构 图片名 图片地址 (上,下)
        # 1. 访问限速问题(一分钟几次限速标准10次/1min，限制并发访问次数)
        apply_api_rate_limit(request_times, max_requests=9, window_seconds=60)  # 限制每分钟最多9次请求
        # 2. 向视觉模型发起请求
        # 2.1 模型对象
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 2.2 准备提示词
        prompt = load_prompt("image_summary", root_folder=stem, image_content=content_data)

        # 图片转成base64
        with open(image_path, "rb") as f:
            # encode返回的是bytes，要转成字符串还需decode，字符->字节->字符
            base64_image = base64.b64encode(f.read()).decode("utf-8")  # b64decode是还原成原始数据

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # 可以放图片网络地址，也可以直接放base64图片的字符串；jpeg对应jpg
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # 2.3 执行获取总结
        response = vm_model.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        summaries[image_file] = summary
        logger.debug(f"图片[{image_file}]总结完成: {summary}")

    logger.info(f"总结图片，获取结果: {summaries}")

    return summaries


def step_4_upload_and_replace_images(targets, summaries, stem, md_content):
    """
    将图片传入到minio服务器，替换原md中的图片和描述
    :param targets: [(图片名, 图片地址, (上文, 下文)), ...]
    :param summaries: {图片名:总结, ...}
    :param stem: md文件名(不带后缀) 文件夹的名称
    :param md_content: 原始md内容
    :return: 替换图片和描述后的md内容
    """

    # minio存储结果：桶/upload-images(env定义)/文件夹名字/图片对象.jpg
    minio_client = get_minio_client()
    # 1. 删除minio中对应文件的图片
    # 1.1 获取要删除的对象
    img_obj_list = minio_client.list_objects(minio_config.bucket_name,
                                             # 去掉第一个斜杠/ 不然会报错
                                             prefix=f"{minio_config.minio_img_dir[1:]}/{stem}/",
                                             recursive=True)
    # 因为obj里面会有元数据，DeleteObject它只看对象名，别的不管
    delete_obj_list = [DeleteObject(obj.object_name) for obj in img_obj_list]
    # 1.2 调用方法进行删除
    errors = minio_client.remove_objects(minio_config.bucket_name, delete_obj_list)
    if errors:  # 如果有错误
        for error in errors:
            logger.warning(error)
    logger.info(f"已删除MinIO中[{minio_config.minio_img_dir}/{stem}/]下所有图片对象，总共{len(delete_obj_list)}个")

    # 2. 上传图片到minio
    # 记录图片上传后的结果
    images_url = {}  # 图片名:url
    for image_file, image_path, _ in targets:  # _是占位符，表示不需要解包出来的参数但是仍需要占位
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                # 桶名后的都叫对象 xx.png  xx/xxx/x.jpg，没有的文件夹会自动创建
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 上传完毕后记录
            # 图片地址：协议+端点+桶名+对象名
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir[1:]}/{stem}/{image_file}"
            logger.info(f"图片上传成功: {image_file} -> {images_url[image_file]}")
        except Exception as e:
            logger.error(f"图片上传失败: {image_file}, 错误: {e}")

    # 3. md图片替换
    # 汇总：{图片名:(描述，url)}
    image_infos = {}
    for image_file, summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = (summary, url)
    logger.info(f"图片处理汇总结果:{image_infos}")

    if image_infos:
        # ![summary](url)，利用正则替换
        for image_file, (summary, url) in image_infos.items():
            # 构建正则表达式，匹配 ![任意内容](任意内容/图片名任意后缀)
            pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(image_file) + r".*?\)")
            # 替换为 ![summary](url)
            md_content = pattern.sub(f"![{summary}]({url})", md_content)
        logger.info("完成替换，新内容为{}".format(md_content))

    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md磁盘备份返回新地址
    :param new_md_content: 新md内容
    :param md_path_obj: 老地址
    :return: 新地址
    """

    # 设置一下新地址
    # splitext(md_path_obj)[0]是取去掉扩展名后的部分
    new_md_path = os.path.splitext(md_path_obj)[0] + "_new.md"  # splitext专门用来分割文件路径的“主名”和“扩展名”

    with open(new_md_path, "w") as f:
        f.write(new_md_content)
    logger.info(f"新md文件已保存: {new_md_path}")

    return new_md_path


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> {function_name} 开始执行; 当前状态为{state}")
    add_running_task(state["task_id"], function_name)

    # 1.校验并且获取本次操作的数据
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)

    # 如果没有图片文件夹,直接返回state
    if not images_dir_obj.exists():
        logger.info(f"图片文件夹不存在: {images_dir_obj}, 跳过图片处理步骤")
        return state
    
    # 2.识别md用过的图片，上文下文各100
    targets = step_2_scan_images(md_content, images_dir_obj)

    # 3.进行图片总结和处理(视觉模型)
    summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)

    # 4. 上传到minio同时替换md图片 描述+url
    new_md_content = step_4_upload_and_replace_images(targets, summaries, md_path_obj.stem, md_content)

    # 5. md内容替换和备份
    new_md_file_path = step_5_replace_md_and_save(new_md_content, md_path_obj)
    # 更新地址内容
    state["md_path"] = new_md_file_path
    state["md_content"] = new_md_content

    logger.info(f">>> {function_name} 执行结束; 当前状态为{state}")
    add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join("output", "hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
