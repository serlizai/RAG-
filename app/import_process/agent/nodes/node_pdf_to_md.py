import os
from pathlib import Path
import sys
import time
import requests
import shutil
import zipfile

from app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.conf.mineru_config import mineru_config

"""
    node_pdf_to_md 
        参数：state [is_pdf_read_enabled = True, local_dir = output, pdf_path = doc/xxx.pdf]
        返回：state [md_path = output/xxx/xxx.md, md_content = "md文件内容"]
        1.日志和任务状态
        2.路径校验
        3.mineru交互
        4.下载解压
        5.日志和任务状态，return state
    validate_paths
        参数：state [pdf_path, local_dir]
        返回：pdf_path_obj Path, local_dir_obj Path
        1.非空校验
        2.文件存在校验 pdf_path_obj没有就抛异常，local_dir_obj没有就给予默认
        3.返回Path对象
    upload_poll
        参数：pdf_path_obj Path
        返回：zip_url str
        1.申请上传url
        2.上传pdf文件  put请求
        3.轮询解析状态  500错误码可容忍,state是列表
        4.拿到下载url
    download_and_extract
        参数：zip_url str, local_dir_obj Path, 原文件名file_title str
        返回：md_path str
        1.下载zip包  get请求
        2.检查地址后解压zip包到指定文件夹并进行防重复处理
        3.重命名
        4.返回md文件绝对路径
"""



def step_1_validate_paths(state: ImportGraphState):
    """
    步骤1: 参数校验，返回校验完成可以直接使用的路径对象
    1. 校验 local_dir，如果没有提供需要有默认值
    2. 校验 local_file_path，确保文件存在，不存在直接异常处理
    """
    logger.debug(f">>> step_1_validate_paths:md转pdf开始进行文件格式校验")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 非空校验
    if not pdf_path:
        logger.error(">>> step_1_validate_paths:pdf_path为空,没有输入文件，无法进行解析")
        raise ValueError("PDF路径为空，无法进行解析")

    if not local_dir:
        local_dir = PROJECT_ROOT / "output"  # / 符号重载，拼接之后是path对象，默认使用项目根目录下的output文件夹
        logger.warning(f">>> step_1_validate_paths:local_dir值为空，使用默认值 '{str(local_dir)}'")

    # 文件存在校验
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)  # Path嵌套Path不会报错，防止传过来的是str和默认的不是一个类型
    if not pdf_path_obj.exists():
        logger.error(f">>> step_1_validate_paths:pdf_path文件不存在: {pdf_path}")
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    if not local_dir_obj.exists():
        logger.warning(f">>> step_1_validate_paths:local_dir文件夹不存在，将创建: {local_dir}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)  # 创建文件夹
    
    return pdf_path_obj, local_dir_obj


def step_2_upload_poll(pdf_path_obj) -> str:
    """
    步骤2: 调用mineru进行pdf解析(local_file_path)返回一个下载文件的url地址
    申请拿到上传url->上传pdf文件->轮询解析状态->拿到下载url
    :param pdf_path_obj: Path对象,上传pdf的文件路径
    :return: 下载zip包的url地址
    """
    # 申请拿到上传url
    # 前置参数：url、token、固定格式请求头
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name":f"{pdf_path_obj.name}"}  # 文件名（含后缀）
        ],
        "model_version":"vlm"
    }
    response = requests.post(url,headers=header,json=data)
    # 检查响应状态码和返回的code字段，确保请求成功
    if response.status_code != 200 or response.json()['code'] != 0:
        logger.error(f">>> step_2_upload_poll:申请上传url失败,状态码: {response.status_code},响应内容: {response.text}")
        raise RuntimeError(f"申请上传url失败,状态码: {response.status_code},响应内容: {response.text}")
    re_json = response.json()
    batch_id = re_json["data"]["batch_id"]
    upload_url = re_json["data"]["file_urls"][0]

    # 文件上传
    # 使用put请求很大概率会报错，put会修改请求头，添加额外的参数头，
    # 将文件存储到文件存储服务器，文件存储服务器查的严格会拒绝，get和post不会这么严格
    http_session = requests.Session()  # 持久会话对象，向同一个服务器发请求不需要反复握手
    http_session.trust_env = False  # 禁用环境代理，防止请求被代理服务器拦截,还能复用请求对象
    
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(upload_url, data=file_data)  # oss预签名URL标准上传方式应该是PUT不是 POST
        if upload_response.status_code != 200:
            logger.error(f">>> step_2_upload_poll:上传文件失败,状态码: {upload_response.status_code}")
            raise RuntimeError(f"step_2_upload_poll上传文件失败,状态码: {upload_response.status_code}")
    except requests.RequestException as e:          # ← 不拦截 RuntimeError！
        # 只处理真正的网络问题：网线断了、DNS 解析失败、超时等
        logger.error(f">>> step_2_upload_poll:上传文件异常: {e}")
        raise RuntimeError(f"step_2_upload_poll上传文件异常: {e}")
    finally:
        http_session.close()  # 关闭会话，释放资源

    # 轮询获取解析结果，因为不一定一次就取到结果
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    # 每3秒轮询一次，最多等待10分钟
    timeout = 600  # 10分钟
    interval = 3   # 3秒轮询一次
    start_time = time.time()  # 记录开始时间
    while True:
        # 超时判断
        if time.time() - start_time > timeout:
            logger.error(f">>> step_2_upload_poll:轮询解析结果超时")
            raise RuntimeError(f"step_2_upload_poll轮询解析结果超时")

        # 向指定url获取本次解析的结果
        res = requests.get(url, headers=header)

        # 解析结果判断和获取zipurl
        if res.status_code != 200:
            # 抛出异常，站在客户端角度只有服务器问题5xx是可以给机会再次请求直到超时
            if 500 <= res.status_code < 600:
                # logger.warning(f">>> step_2_upload_poll:轮询解析结果服务器错误,状态码: {res.status_code}")
                time.sleep(interval)
                continue
            raise RuntimeError(f"step_2_upload_poll轮询解析结果失败,状态码: {res.status_code}")
        
        json_data = res.json()
        if json_data['code'] != 0:
            # logger.error(f">>> step_2_upload_poll:轮询解析结果失败,响应内容: {res.text}")
            raise RuntimeError(f"step_2_upload_poll轮询解析结果失败")
        
        # 解析状态判断
        status = json_data['data']['extract_result'][0]
        if status['state'] == 'done':
            zip_url = status['full_zip_url']
            logger.info(f">>> pdf解析成功,耗时: {time.time() - start_time:.2f}秒,结果: {zip_url}")
            return zip_url
        else:
            time.sleep(interval)  # 解析未完成，等待一段时间后继续轮询


def step_3_download_and_extract(zip_url: str, local_dir_obj: Path, file_title) -> str:
    """
    下载指定的md.zip文件并解压，返回md文件存放路径
    :param zip_url: 下载zip包的url地址
    :param local_dir_obj: Path对象,解压的文件夹
    :param file_title: 文件名(用来命名输出的文件夹)
    :return: 解压后的md文件路径
    """
    # 下载zip包
    zip_response = requests.get(zip_url)
    if zip_response.status_code != 200:
        logger.error(f">>> step_3_download_and_extract:下载zip包失败,状态码: {zip_response.status_code}")
        raise RuntimeError(f"step_3_download_and_extract下载zip包失败,状态码: {zip_response.status_code}")

    # 保存zip包到本地 路径：output/文件名.zip
    zip_path = local_dir_obj / f"{file_title}.zip"
    # 把从服务器下载到的 zip 原始字节原封不动写进open创建的空zip压缩包里
    with open(zip_path, 'wb') as f:
        f.write(zip_response.content)
    logger.info(f">>> step_3_download_and_extract:下载zip包成功,保存路径: {zip_path}")

    # 清空旧目录，因为两次解压的文件数量可能不一样，旧数据会部分保留干扰新数据
    extracted_dir = local_dir_obj / file_title
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)  # 这个文件夹也会删除
        logger.info(f">>> step_3_download_and_extract:清空旧目录: {extracted_dir}")
    extracted_dir.mkdir(parents=True, exist_ok=True)  # 创建新目录

    # 解压zip包
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:  # r是只能读取解压
        zip_ref.extractall(extracted_dir)  # 解压到 output/文件名/ 下

    # 删除zip包，节省空间
    # zip_path.unlink()

    # 返回解压后的md文件路径，解压后的文件名可能是文件.md，低版本是full.md
    # 惰性生成器是用到才去计算
    md_file_list = list(extracted_dir.rglob("*.md"))  # rglob是当前目录 + 所有子目录，glob是当前目录，返回惰性生成器
    target_md_file = None  # 存储最终md文件

    if not md_file_list:
        logger.error(f">>> step_3_download_and_extract:解压后未找到md文件: {extracted_dir}")
        raise FileNotFoundError(f"解压后未找到md文件: {extracted_dir}")

    # 检查有没有原文件名的md
    for md_file in md_file_list:
        if md_file.name == file_title + ".md":
            target_md_file = md_file
            break

    # 如果没有原文件名的md就找full.md
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
    
    # 如果还是没有就取第一个md文件
    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.warning(f">>> step_3_download_and_extract:未找到原文件名或full.md，使用第一个md文件: {target_md_file}")

    # 重命名解压后的md文件为原文件名.md，方便后续处理，不是原名字才重命名
    if target_md_file.stem != file_title:
        # with_name它只替换路径最后一级的文件名（含后缀），前面的目录部分保持不变。
        # 它不检查文件是否真实存在，只是做纯字符串路径的替换，不涉及文件操作∫
        # 返回的是新对象，原 Path 对象不会被修改
        # rename才真正修改磁盘文件名并返回新的path
        target_md_file = target_md_file.rename(target_md_file.with_name(file_title + ".md"))

    logger.info(f">>> step_3_download_and_extract:重命名md文件为原文件名: {target_md_file}")
    return str(target_md_file.resolve())  # 返回md文件绝对路径字符串路径，resolve()返回绝对路径


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    进入和结束的日志和任务状态的配置
    参数校验 local_dir没有的话要有默认值 local_file_path完成字面意思的校验，深入校验文件是否真的存在
    1. 调用 MinerU (magic-pdf) 工具，返回一个下载文件的url地址
    2. 将 PDF 转换成 Markdown 格式。
    3. 根据返回的下载文件的url地址，下载zip包，赋值md_path
    4.将结果保存到 state["md_content"]。
    这里需要try-except，因为涉及到了第三方
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> {function_name} 开始执行; 当前状态为{state}")
    add_running_task(state["task_id"], function_name)  # 用于在前端实时展示当前工作流的执行进度

    try:
        # 参数校验，返回校验完成可以直接使用的路径对象
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
        # 调用mineru进行pdf解析(local_file_path)返回一个下载文件的url地址
        # 参数是要解析的pdf文件路径
        zip_url = step_2_upload_poll(pdf_path_obj)
        # 下载zip包并解压提取，返回md_path
        # 参数：要下载的地址 解压的文件夹 文件名(用来命名输出的文件夹)
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
        # 更新数据
        state["md_path"] = md_path
        state['local_dir'] = str(local_dir_obj)
        # 读取md文件内容，保存到state["md_content"]
        with open(md_path, 'r', encoding='utf-8') as f:
            state["md_content"] = f.read()
    except Exception as e:
        # 处理异常，记录日志并更新状态
        logger.error(f">>> [{function_name}] 执行MinerU解析异常: {e}")
        raise # TODO: 这里直接抛出异常，终止工作流，后续可以考虑更优雅的处理方式
    finally:
        # 结束的日志和任务状态的配置
        logger.info(f">>> [{function_name}] 执行结束; 当前状态为{state}")  
        add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")