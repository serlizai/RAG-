import re
import json
import os
import sys
from email.mime import text
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

"""
    md内容切块
    最终:chunks -> 存储切块的集合  chunks -> 备份到本地 -> chunks.json
    1. 参数校验(材料是否完整)
    2. 粗粒度切割md  -> 使用标题切割  保证语义完整
    3. 特殊情况:一个文档没有标题，给一个默认标题(兜底)
    4. 细粒度切割 块大小和重叠:大->小(同时设置重叠) 小->大(合并) 得到大小合适语义完整的chunks
    5. 数据备份和修改chunks属性(本地备份、chunks存入state)
    return state
"""



def step_1_get_content(state):
    # 读取要切片的内容
    md_content = state['md_content']
    if not md_content:
        logger.error("[step_1_get_content]没有有效的md内容，直接抛出异常")
        raise ValueError("没有有效的md内容，无法进行切分")
    # 处理md中的换行符
    """
        windows \r\n
        mac/linux \n
        老mac \r
    """
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get('file_title', 'default_title')

    return md_content, file_title


def step_2_split_by_title(md_content, file_title):
    """
    根据标题进行语义切割
    :param md_content:
    :param file_title:
    :return: [{content, title, file_title},{}]
    """
    """
    标题：#->######[空格]标题名
    代码块可能也有#注释
    什么时候创建->{content, title, file_title}->标题、不能是代码块
    ## 开篇
    内容\n
    ![]()
    内容\n
    ## 中篇
    内容\n
    ## 下篇
    内容\n
    """

    # 1.前置准备
    # 1.1 正则 \s* 0个或多个空白字符  \s+至少一个空白字符  .+至少一个除换行外的任意字符  ^ 从行首开始
    title_pattern = r'^\s*#{1,6}\s+.+'
    # 1.2 md_content切割\n
    lines = md_content.split("\n")  # list[str]
    # 1.3 定义临时存储变量  current_title=str|current_lines=[]|title_count=0 存储了多少块|is_code_block 是否是代码块
    current_title = ""
    current_lines = []  # 当前标题行
    title_count = 0
    is_code_block = False
    # 1.4 最终存储的列表 sections=[]
    sections = []

    def _flush_section():
        """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
        if not current_lines:
            return
        sections.append({
            "title": current_title,  # 切片标题
            # 每段时间使用 \n换行区分
            "content": "\n".join(current_lines),
            "file_title": file_title,  # 文档名
        })


    # 2.循环每行的列表
    for line in lines:
        strip_line = line.strip()  # 去除首尾空白字符防止匹配不上
        # 2.1判断代码块状态
        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            # 进入或者退出代码块
            is_code_block = not is_code_block  # 取反
            # 内容肯定不是标题
            current_lines.append(line)
            continue
        # 2.2判断是不是标题
        is_title = re.match(title_pattern, line) and not is_code_block
        # 2.3是标题怎么处理
        # 如果不要空标题：current_title not null and len(current_lines) > 1
        if is_title:
            # 检查是不是第一次标题，不是的话就存储
            _flush_section()
            current_title = strip_line
            current_lines = [current_title]  # 指向一个全新的列表对象
            title_count += 1
            logger.debug(f"识别到标题:{current_title}")
        # 2.4不是标题怎么处理
        else:
            current_lines.append(line)
    _flush_section()
    logger.info(f"步骤2：MD标题切分完成，识别到{title_count}个有效标题，原始文本共{len(lines)}行")

    # 3.返回结果sections
    return sections, title_count, len(lines)


def step_3_handle_no_title(sections, md_content, file_title, title_count):
    """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param md_content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
    if title_count == 0:
        # 无标题情况：替换为单章节，标题为"无标题"
        logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
        return [{"title": "无标题", "content": md_content, "file_title": file_title}]
    # 有标题情况：直接返回步骤2的结果
    logger.debug(f"步骤3：检测到{title_count}个有效标题，无需兜底处理")
    return sections


def split_long_section(section, max_length):
    """
    将当前chunk内容进行二次切割
    :param section:
    :param max_length:
    :return: [{title, content, file_title, parent_title, part},{},...]
    """
    # 获取content
    content = section.get("content")
    # 判断content是否超长，没有就直接返回
    if len(content) <= max_length:
        logger.info(f"当前块长度{len(content)}未超过最大长度{max_length}，无需二次切割")
        return [section]
    # 超长进行二次切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,  # 切块长度，不可能大于这个值
        chunk_overlap=100,  # 重叠长度
        separators=["\n\n", "\n", " ", ""]  # 切割符号
    )

    # title = 标题名_1/2/3  part 1 2 3 | parent_title = sections.title
    sub_sections = []
    for index, chunk in enumerate(splitter.split_text(content), start=1):
        text = chunk.strip()
        title = f"{section.get('title')}_{index}"
        parent_title = section.get("title")
        part = index
        file_title = section.get("file_title")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": parent_title,
            "part": part
        })

    # 返回切割后的结果
    return sub_sections


def merge_short_sections(final_sections, min_length = MIN_CONTENT_LENGTH):
    """
    切的太碎需要合并(双指针处理)
        1.content长度小于min_length
        2.同一个parent_title才能合并
    :param final_sections:
    :param min_length:
    :return: merged sections
    """

    # 边界处理：空列表直接返回，避免后续索引报错
    if not final_sections:
        logger.debug("待合并Chunk列表为空，直接返回")
        return []

    merge_sections = []  # 存储合并结果
    pre_section = None  # 上一次处理的块

    for section in final_sections:
        if pre_section is None:
            pre_section = section  # 处理第一次的块赋值
            continue

        # pre_section是上一次 section是当前块
        is_short = len(pre_section.get("content")) < min_length  # 判定是否小于最小值
        # 判断是否是同一个父标题,且防止没切割过出现None==None
        is_same_parent = section.get("parent_title") and section.get("parent_title") == pre_section.get("parent_title")

        if is_short and is_same_parent:
            # 上一次是短块和本次进行合并
            current_content = section.get("content")
            pre_section['content'] += "\n\n" + current_content
            pre_section['part'] = section.get('part')  # 改成最新的part
            logger.debug(f"合并短Chunk：{pre_section.get('parent_title')} → 累计长度{len(pre_section['content'])}")
        else:
            # 不是短块或父标题不同
            merge_sections.append(section)
            pre_section = section  # 迭代

    # 最后一个块可能没有被加入merge_sections，需要检查
    if pre_section is not None:
        merge_sections.append(pre_section)

    logger.info(f"短Chunk合并完成：原{len(final_sections)}个 → 合并后{len(merge_sections)}个")
    return merge_sections


def step_4_refine_chunks(sections,
                         max_length=  DEFAULT_MAX_CONTENT_LENGTH,
                         min_length = MIN_CONTENT_LENGTH):
    """
    精细切割分块：
        1.小于MIN_CONTENT_LENGTH做合并，前提是同一个parent_title
        2.大于DEFAULT_MAX_CONTENT_LENGTH做切割(parent_title｜part)
    :param sections:
    :param max_length: chunks最大长度
    :param min_length: chunks最小长度
    :return: sections
    """
    # 边界处理：最大长度无效（空/≤0），直接返回原章节，避免切分异常
    if not max_length or max_length <= 0 or not min_length or min_length <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_length}），跳过精细化处理")
        return sections

    final_sections = []  # 存储处理之后的块

    # 超过的先切碎
    for section in sections:
        # section: {title, content, file_title}
        # 切块变成:[{title, content, file_title, parent_title, part},{},...]
        sub_section = split_long_section(section, max_length)
        # append会将整个列表加进去会嵌套，extend是将元素逐个加进去
        final_sections.extend(sub_section)
    # 小于的再合并
    final_sections = merge_short_sections(final_sections, min_length)
    # 补全属性和参数 粗切合格的块会没有parent_title和part，存进向量数据库会报错
    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title')
    # 返回结果
    return final_sections


def step_5_backup_chunks(state, sections):
    """
    存储备份切块
    :param state: 本地地址local_dir
    :param sections: 存储的切块
    :return:
    """
    local_dir = state.get('local_dir')
    if not local_dir:
        logger.warning("步骤6：未配置备份目录（local_dir），跳过Chunk结果备份")
        return
    # 创建备份目录：已存在则不报错（exist_ok=True）
    try:
        os.makedirs(local_dir, exist_ok=True)
        backup_dir = os.path.join(local_dir, 'chunks.json')

        with open(backup_dir, 'w', encoding="utf-8") as f:
            """
                sections是Python 嵌套数据结构（List[Dict[str, Any]]，列表里装字典，字典里可能嵌套字符串 / 数字等），
                而普通文件写入（如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，
                无需手动转换为字符串，同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
            """
            json.dump(
                sections,  # 要写的数据
                f,   # 要写的位置
                # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                ensure_ascii=False,  # 保留中文，不转义为\u编码
                indent=2  # 缩进为2
            )
        logger.info(f"步骤5：切块备份完成，已保存到 {backup_dir}")
    except Exception as e:
        # 备份失败仅记录日志，不终止主流程
        logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)


def print_status(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
    输出文档切分统计信息（日志记录，便于监控/调试）
    :param lines_count: MD原始文本总行数
    :param sections: 最终处理后的Chunk列表
    """
    chunk_num = len(sections)
    # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
    logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
    logger.info(f"MD原始文本总行数：{lines_count}")
    logger.info(f"最终生成Chunk数量：{chunk_num}")
    if sections:
        first_title = sections[0].get("title", "无标题")
        logger.info(f"首个Chunk标题预览：{first_title}")
    logger.info("-" * 110)


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}")
    add_running_task(state['task_id'], function_name)

    try:
        # 1.参数校验
        md_content, file_title = step_1_get_content(state)

        # 2.粗粒度切割md  -> 使用标题切割
        # [{content:标题内容, title:标题, file_title:文件名},{},...]
        sections, title_count, lines_count = step_2_split_by_title(md_content, file_title)

        # 3.特殊情况: 一个文档没有标题，给一个默认标题(兜底)
        sections = step_3_handle_no_title(sections, md_content, file_title, title_count)
        # 4.细粒度切割 块大小和重叠: 大->小(同时设置重叠) 小->大(合并) 得到大小合适语义完整的chunks
        sections = step_4_refine_chunks(sections, DEFAULT_MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH)
        # 输出切分统计信息
        print_status(lines_count, sections)
        # 5.数据备份和修改chunks属性(本地备份、chunks存入state)
        state['chunks'] = sections
        step_5_backup_chunks(state, sections)
    except Exception as e:
        logger.error(f">>>[{function_name}] 执行出错: {e}")
        raise  # 终止工作流
    finally:
        logger.info(f">>> [Stub] 完成节点: {function_name},当前状态为{state}")
        add_done_task(state['task_id'], function_name)

    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")