from ultralytics import YOLO
from screen_to_world_locator import world_position_from_depth, screen_distance_to_world_distance
from typing import List, Any, Dict
import requests
import json
import re
import ast
import easyocr
import cv2
import os
import numpy as np
import tempfile
import shutil
try:
    import OpenEXR  # type: ignore
    import Imath  # type: ignore
except Exception:
    OpenEXR = None  # type: ignore
    Imath = None  # type: ignore
from agent_memory.user_info.manager import DBManager as UserInfoManager
from agent_memory.prompt_manager.scene_iteams.manager import DBManager as SceneItemEntryManager

# --------------- 通用工具 ---------------
def build_rts_matrix(translation: List[float], rotation_deg: List[float], scale: List[float]) -> np.ndarray:
    """
    构建 4x4 的旋转-平移-缩放变换矩阵。

    Args:
        translation: [tx, ty, tz]
        rotation_deg: [rx_deg, ry_deg, rz_deg] 依次为绕 X、Y、Z 轴的欧拉角(度)，右手坐标系，按 Z·Y·X 组合
        scale: [sx, sy, sz]

    Returns:
        形如 (4,4) 的 numpy 矩阵，等于 T @ R @ S
    """
    tx, ty, tz = float(translation[0]), float(translation[1]), float(translation[2])
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    rx, ry, rz = np.deg2rad([rotation_deg[0], rotation_deg[1], rotation_deg[2]])

    # 绕 X 轴旋转
    cx, sxn = np.cos(rx), np.sin(rx)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sxn],
        [0.0, sxn, cx]
    ], dtype=np.float64)

    # 绕 Y 轴旋转
    cy, syn = np.cos(ry), np.sin(ry)
    Ry = np.array([
        [cy, 0.0, syn],
        [0.0, 1.0, 0.0],
        [-syn, 0.0, cy]
    ], dtype=np.float64)

    # 绕 Z 轴旋转
    cz, szn = np.cos(rz), np.sin(rz)
    Rz = np.array([
        [cz, -szn, 0.0],
        [szn, cz, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 组合旋转：Z · Y · X（常用的航向-俯仰-横滚顺序）
    R3 = Rz @ Ry @ Rx

    # 嵌入到 4x4
    R4 = np.eye(4, dtype=np.float64)
    R4[:3, :3] = R3

    # 缩放矩阵（各向异性）
    S4 = np.eye(4, dtype=np.float64)
    S4[0, 0] = sx
    S4[1, 1] = sy
    S4[2, 2] = sz

    # 平移矩阵
    T4 = np.eye(4, dtype=np.float64)
    T4[:3, 3] = [tx, ty, tz]

    # 最终变换：先缩放，再旋转，最后平移
    M = T4 @ (R4 @ S4)
    return M
def _extract_and_parse_json_array(text: str):
    """
    从包含 Markdown 代码块或纯文本的字符串中提取并解析 JSON 数组。
    - 优先提取 ```json ... ``` 或 ``` ... ``` 代码块内的内容
    - 失败则尝试在全文中寻找首个以 [ 开始、] 结束的平衡数组
    - 解析时先 json.loads，失败则进行常见纠错（中英文引号、单引号、尾随逗号）后再尝试
    返回: list 或引发异常
    """
    if not isinstance(text, str) or not text:
        raise ValueError("空响应，无法解析JSON")

    content = text

    # 1) 优先提取 ```json ... ``` 代码块
    m = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
    if not m:
        # 2) 其次提取任意 ``` ... ``` 代码块
        m = re.search(r"```\s*([\s\S]*?)\s*```", content)
    if m:
        candidate = m.group(1)
    else:
        # 3) 无代码块时，尝试在全文中寻找首个平衡的 JSON 数组
        #    从第一个 '[' 开始向后扫描，匹配到成对的 ']' 结束
        start = content.find('[')
        if start == -1:
            raise ValueError("未找到JSON数组起始标记 [ ")
        depth = 0
        end = -1
        for i, ch in enumerate(content[start:], start=start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            raise ValueError("未找到匹配的 ] 以结束JSON数组")
        candidate = content[start:end+1]

    # 标准化常见问题
    s = candidate.strip()
    # 直接删除中文/智能引号字符，避免被当成 JSON 定界符导致解析错误
    # 保留 ASCII 引号 " 和 ' 不变
    s = s.replace('“', '').replace('”', '').replace('‘', '').replace('’', '')

    def _remove_trailing_commas(x: str) -> str:
        # 移除对象或数组末尾多余逗号: ,}\n 或 ,]\n
        x = re.sub(r",\s*}\s*", "}", x)
        x = re.sub(r",\s*]\s*", "]", x)
        return x

    # 第一轮: 直接尝试严格JSON解析
    try:
        return json.loads(s)
    except Exception:
        pass

    # 第二轮: 移除尾随逗号
    s2 = _remove_trailing_commas(s)
    try:
        return json.loads(s2)
    except Exception:
        pass

    # 第三轮: 将单引号对象转成双引号（避免替换字符串内部合法内容，先粗略处理）
    # 仅当看起来是 Python 风格的 dict/list 时才尝试
    looks_like_py = re.search(r"[\{\[]\s*'", s2) or re.search(r"'\s*:\s*", s2)
    s3 = s2
    if looks_like_py:
        # 简单将键与字符串值的引号替换为双引号
        # 注意：这不是完美方案，但对常见 LLM 输出足够鲁棒
        # 先避免替换数字/布尔/null
        # 采用一个温和的替换：仅替换包围在引号中的成对单引号
        s3 = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', s2)
    try:
        return json.loads(s3)
    except Exception:
        pass

    # 第四轮: 兜底使用 ast.literal_eval 尝试 Python 字面量
    try:
        return ast.literal_eval(s3)
    except Exception as e:
        # 仍失败则抛错并带上片段用于定位问题
        snippet = s3[:500]
        raise ValueError(f"JSON解析失败: {e}. 片段: {snippet}")

# 初始化EasyOCR，支持中文和英文（可指定模型目录）
_easyocr_model_dir = os.environ.get("EASYOCR_MODEL_DIR", "./easyocr_model")
if _easyocr_model_dir and os.path.isdir(_easyocr_model_dir):
    print(f"使用自定义 EasyOCR 模型目录: {_easyocr_model_dir}")
    reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory=_easyocr_model_dir, download_enabled=False)
else:
    reader = easyocr.Reader(['ch_sim', 'en'])

def fast_ocr(img_input):
    """
    使用EasyOCR识别图片中的文字，返回文字内容和位置信息
    
    Returns:
        List[Dict]: 包含文字内容和边界框信息的列表
    """
    # 支持传入路径或内存中的图像(ndarray)
    if isinstance(img_input, np.ndarray):
        img = img_input
    else:
        img = cv2.imread(img_input)
    original_h, original_w = img.shape[:2]
    
    # 可选：缩放图像到 800px 宽度以内加速
    scale_factor = 1.0
    if original_w > 800:
        scale_factor = 800 / original_w
        img = cv2.resize(img, (800, int(800 * original_h / original_w)))
    
    # EasyOCR返回格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence]
    result = reader.readtext(img)
    
    # 调试信息：打印OCR返回的数据结构
    print(f"EasyOCR返回结果类型: {type(result)}")
    print(f"EasyOCR结果数量: {len(result)}")
    if result:
        print(f"第一个结果示例: {result[0]}")
    
    ocr_results = []
    for i, detection in enumerate(result):
        try:
            # EasyOCR返回格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence]
            points = detection[0]  # 四个点的坐标
            text_content = detection[1]  # 文字内容
            confidence = detection[2]  # 置信度
            
            print(f"处理OCR结果 {i}: 文字='{text_content}', 置信度={confidence:.3f}")
            
            # 计算边界框
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            
            # 如果图像被缩放，需要将坐标还原到原始尺寸
            if scale_factor != 1.0:
                x1 = x1 / scale_factor
                y1 = y1 / scale_factor
                x2 = x2 / scale_factor
                y2 = y2 / scale_factor
            
            ocr_results.append({
                'text': text_content,
                'confidence': confidence,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'type': 'text'  # 标记为文字类型
            })
        except Exception as e:
            print(f"处理OCR结果时出错: {e}, detection={detection}")
            continue
    
    return ocr_results

def cluster_ocr_results(ocr_results, distance_threshold=50):
    """
    将OCR结果按位置聚类，相近位置的文字合并为一个对象
    
    Args:
        ocr_results: OCR识别结果列表
        distance_threshold: 聚类距离阈值（像素）
    
    Returns:
        List[Dict]: 聚类后的OCR结果
    """
    if not ocr_results:
        return []
    
    clusters = []
    
    for ocr_item in ocr_results:
        # 计算当前文字的中心点
        center_x = (ocr_item['x1'] + ocr_item['x2']) / 2
        center_y = (ocr_item['y1'] + ocr_item['y2']) / 2
        
        # 寻找最近的聚类
        closest_cluster = None
        min_distance = float('inf')
        
        for i, cluster in enumerate(clusters):
            # 计算到聚类中心的距离
            cluster_center_x = (cluster['x1'] + cluster['x2']) / 2
            cluster_center_y = (cluster['y1'] + cluster['y2']) / 2
            
            distance = ((center_x - cluster_center_x) ** 2 + (center_y - cluster_center_y) ** 2) ** 0.5
            
            if distance < distance_threshold and distance < min_distance:
                min_distance = distance
                closest_cluster = i
        
        if closest_cluster is not None:
            # 合并到现有聚类
            cluster = clusters[closest_cluster]
            cluster['texts'].append(ocr_item['text'])
            cluster['confidences'].append(ocr_item['confidence'])
            
            # 更新边界框
            cluster['x1'] = min(cluster['x1'], ocr_item['x1'])
            cluster['y1'] = min(cluster['y1'], ocr_item['y1'])
            cluster['x2'] = max(cluster['x2'], ocr_item['x2'])
            cluster['y2'] = max(cluster['y2'], ocr_item['y2'])
            
            # 更新平均置信度
            cluster['confidence'] = sum(cluster['confidences']) / len(cluster['confidences'])
            
            # 更新合并后的文字内容
            cluster['text'] = ' '.join(cluster['texts'])
        else:
            # 创建新的聚类
            new_cluster = {
                'text': ocr_item['text'],
                'confidence': ocr_item['confidence'],
                'x1': ocr_item['x1'],
                'y1': ocr_item['y1'],
                'x2': ocr_item['x2'],
                'y2': ocr_item['y2'],
                'type': 'text',
                'texts': [ocr_item['text']],  # 存储所有文字
                'confidences': [ocr_item['confidence']]  # 存储所有置信度
            }
            clusters.append(new_cluster)
    
    # 清理临时字段，只保留最终结果需要的字段
    for cluster in clusters:
        del cluster['texts']
        del cluster['confidences']
    
    return clusters

# 导入数据库相关模块
from agent_memory.prompt_manager.scene_iteams.manager import DBManager as SceneItemsManager

class ChatModel():
    def __init__(self, model_name: str, model_provider: str, api_key: str, api_base: str):
        self.model_name = model_name
        self.model_provider = model_provider
        self.api_key = api_key
        self.api_base = api_base
    
    def invoke(self, messages: List[Any], extra_body: Dict[str, Any] = None) -> Any:
        """
        基于ARK API的同步聊天方法，替代原有的chat_model.invoke

        Args:
            messages: 消息列表
            extra_body: 额外参数（可选）

        Returns:
            聊天接口返回的内容
        """
        # 构建请求数据
        payload = {
            "messages": [],
            "model": self.model_name,
            "stream": False
        }

        # 转换消息格式
        for message in messages:
            if isinstance(message, dict):
                # 直接是字典
                if message.get('role', None) and message.get('content', None):
                    payload["messages"].append({
                        "role": message.get('role'),
                        "content": message.get('content')
                    })
                else:
                    payload["messages"].append(message)
            else:
                # 兼容langchain的消息对象
                if hasattr(message, "type") and hasattr(message, "content"):
                    payload["messages"].append({
                        "role": getattr(message, "type"),
                        "content": getattr(message, "content")
                    })

        # 添加额外参数
        if extra_body:
            payload.update(extra_body)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            if response.status_code != 200:
                raise Exception(f"ARK API请求失败: {response.status_code} - {response.text}")
            result = response.json()
            # 兼容ARK返回格式
            if "choices" in result and len(result["choices"]) > 0:
                # 返回第一个choice的message内容
                return type("ChatResponse", (), {"content": result["choices"][0]["message"]["content"]})()
            return result
        except Exception as e:
            raise Exception(f"invoke请求异常: {e}")

class EmbeddingModel():
    def __init__(self, model_name: str, api_key: str, api_base: str):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

    def embed(self, input_text: str) -> List[float]:
        try:
            payload = {
                "input": input_text,
                "model": self.model_name
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            resp = requests.post(f"{self.api_base}/embeddings", headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"嵌入请求失败: {resp.status_code} - {resp.text}")
                return []
            data = resp.json()
            emb = ((data or {}).get("data") or [{}])[0].get("embedding", [])
            
            # 确保是浮点数组
            if isinstance(emb, list):
                try:
                    emb_array = np.array(emb, dtype=float)
                    # numpy normalize
                    emb_normalized = emb_array / np.linalg.norm(emb_array)
                    # 取前1024个元素
                    emb_truncated = emb_normalized[:1024]
                    # 转换为列表
                    return emb_truncated.tolist()
                except Exception:
                    return []
            return []
        except Exception as e:
            print(f"获取文本嵌入异常: {e}")
            return []

def decode_exr_from_bytes_via_tempfile(exr_bytes: bytes) -> np.ndarray:
    """
    将字节写入临时 .exr 文件后使用 OpenEXR + Imath 读取。
    返回 float64 的单通道深度矩阵；通道优先顺序：Z > R > Y > 任意。
    """
    if OpenEXR is None or Imath is None:
        raise RuntimeError("OpenEXR/Imath 未安装，无法解析 EXR。")

    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmpf:
        tmp_path = tmpf.name
        tmpf.write(exr_bytes)
        tmpf.flush()

    try:
        exr_file = OpenEXR.InputFile(tmp_path)
        header = exr_file.header()
        dw = header['dataWindow']
        h = dw.max.y - dw.min.y + 1
        w = dw.max.x - dw.min.x + 1

        # 按你之前的优先级顺序尝试通道
        channels_to_try = ['Y', 'Z', 'R', 'VIEW_Z', 'DEPTH']
        available = set(header['channels'].keys())
        chosen = None
        arr = None
        for ch in channels_to_try:
            if ch not in available:
                continue
            # 先尝试 FLOAT，再尝试 HALF
            try:
                data_str = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                arr = np.frombuffer(data_str, dtype=np.float32)
            except Exception:
                try:
                    data_str = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.HALF))
                    arr = np.frombuffer(data_str, dtype=np.float16).astype(np.float32)
                except Exception:
                    arr = None
            if arr is not None and arr.size == h * w:
                chosen = ch
                break

        # 如果上述优先级都未成功，尝试任意可用通道
        if arr is None:
            for ch in available:
                try:
                    data_str = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                    arr = np.frombuffer(data_str, dtype=np.float32)
                except Exception:
                    try:
                        data_str = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.HALF))
                        arr = np.frombuffer(data_str, dtype=np.float16).astype(np.float32)
                    except Exception:
                        arr = None
                if arr is not None and arr.size == h * w:
                    chosen = ch
                    break

        if arr is None:
            raise RuntimeError(f"未找到有效深度通道，尝试: {channels_to_try}")

        arr = arr.reshape((h, w)).astype(np.float64)
        print(f"depth.exr 读取成功，通道 {chosen}，shape={arr.shape}")
        return arr
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def correct_ocr_with_vl(chat_model, image_path: str, ocr_results: List[Dict]) -> List[Dict]:
    """
    使用VL模型重新整合OCR识别结果
    
    Args:
        chat_model: 聊天模型实例
        image_path: 图片路径
        ocr_results: OCR识别结果列表
        
    Returns:
        重新整合后的OCR结果列表
    """
    # 构建OCR结果的描述
    ocr_desc = "OCR识别到的文字:\n"
    for i, ocr in enumerate(ocr_results):
        ocr_desc += f"{i+1}. 位置: ({ocr['x1']:.1f}, {ocr['y1']:.1f}) 到 ({ocr['x2']:.1f}, {ocr['y2']:.1f}), 置信度: {ocr['conf']:.3f}, 文字: '{ocr['text_content']}'\n"
    
    correction_prompt = f"""
你是一个专业的图像识别专家。请仔细查看图片，并重新整合OCR模型的识别结果。

{ocr_desc}

请根据图片中的实际内容，重新整合这些OCR识别结果：
1. 合并应该在一起的文字（如完整的句子、短语、标识等）
2. 修正错误的文字识别
3. 过滤掉无意义的识别结果
4. 提供更准确的文字内容和位置

## 输出格式
请按照以下JSON格式输出重新整合结果：
```json
[
  {{'index': 1, 'text': '正确的文字内容', 'x1': 边界框左上角X, 'y1': 边界框左上角Y, 'x2': 边界框右下角X, 'y2': 边界框右下角Y, 'confidence': 'high/medium/low', 'reason': '整合原因'}},
  {{'index': 2, 'text': '正确的文字内容', 'x1': 边界框左上角X, 'y1': 边界框左上角Y, 'x2': 边界框右下角X, 'y2': 边界框右下角Y, 'confidence': 'high/medium/low', 'reason': '整合原因'}}
]
```

请确保：
1. 仔细分析图片中的每个文字区域
2. 提供准确的中文或英文文字内容
3. 给出合理的边界框坐标
4. 如果原识别正确，保持原结果
5. 给出整合的置信度和原因
6. 过滤掉的识别结果不要输出
"""
    
    messages = [{
        "role": "system",
        "content": correction_prompt
    }, {
        "role": "user",
        "content": [
            {
                "image_url": {"url": image_path},
                "type": "image_url"
            },
            {
                "text": "请重新整合上述OCR识别结果",
                "type": "text"
            }
        ]
    }]
    
    try:
        response = chat_model.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
        print(f"VL模型OCR整合响应: {response.content}")
        
        # 解析JSON响应（更健壮）
        try:
            corrections = _extract_and_parse_json_array(response.content)
        except Exception as pe:
            print(f"VL模型OCR整合JSON解析失败: {pe}")
            return []

        # 转换为与YOLO结果相同格式的OCR结果
        ocr_results = []
        for correction in corrections:
            conf_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            conf_val = conf_map.get(str(correction.get('confidence', 'medium')).lower(), 0.6)
            ocr_result = {
                'index': len(ocr_results) + 1,
                'x1': correction['x1'],
                'y1': correction['y1'],
                'x2': correction['x2'],
                'y2': correction['y2'],
                'conf': conf_val,
                'class_id': -1,
                'class_name': 'text',
                'corrected_class': correction.get('text', ''),
                'desc': f"文字: {correction.get('text', '')}",
                'text_content': correction.get('text', ''),
                'type': 'text',
                'correction_confidence': correction.get('confidence', ''),
                'correction_reason': correction.get('reason', '')
            }
            ocr_results.append(ocr_result)

        return ocr_results
    except Exception as e:
        print(f"VL模型OCR整合失败: {e}")
        return []

def correct_detection_with_vl(chat_model, image_path: str, detection_results: List[Dict]) -> List[Dict]:
    """
    使用VL模型修正YOLO检测结果的类别
    
    Args:
        chat_model: 聊天模型实例
        image_path: 图片路径
        detection_results: YOLO检测结果列表
        
    Returns:
        修正后的检测结果列表
    """
    # 构建检测结果的描述
    detection_desc = "YOLO检测到的对象:\n"
    for i, det in enumerate(detection_results):
        detection_desc += f"{i+1}. 当前类别: {det['class_name']}\n"
    
    correction_prompt = f"""
你是一个专业的图像识别专家。请仔细查看图片，并修正YOLO模型的检测结果。

{detection_desc}

请根据图片中的实际内容，为每个检测到的对象提供**最符合画面内容的、合理的类别名称修正**。即使原检测对象在画面中不存在，也请**尽力在画面中寻找最相似或最可能被误检为该物体的实体**，并将其修正为正确的类别名称。

## 重要原则
- **禁止输出“无”、“不存在”、“未检测到”等否定性类别名**。
- 即使原检测是误检，也必须找到画面中最接近该物体的实体，给出合理的替代类别。
- 修正应基于视觉相似性、上下文语义、物体常见误检模式等进行合理推断。
- 若原类别正确，则保持不变。

## 输出格式
请按照以下JSON格式输出修正结果：
```json
[
  {{'index': 1, 'corrected_class': '正确的类别名称', 'desc': '详细的对象描述', 'confidence': 'high/medium/low', 'reason': '修正原因'}},
  {{'index': 2, 'corrected_class': '正确的类别名称', 'desc': '详细的对象描述', 'confidence': 'high/medium/low', 'reason': '修正原因'}}
]
```

## 输出要求
1. 仔细分析图片中的每个检测区域
2. 提供准确的中文类别名称
3. 为每个对象提供详细的外观描述（颜色、形状、材质、位置等）
4. 如果原检测正确，保持原类别
5. 给出修正的置信度和原因
6. **绝对禁止输出“无”或否定性修正结果** —— 必须提供一个积极、合理的替代类别
"""
    
    messages = [{
        "role": "system",
        "content": correction_prompt
    }, {
        "role": "user",
        "content": [
            {
                "image_url": {"url": image_path},
                "type": "image_url"
            },
            {
                "text": "请修正上述检测结果",
                "type": "text"
            }
        ]
    }]

    # messages = [{
    #     "role": "user",
    #     "content": [
    #         {
    #             "image_url": {"url": image_path},
    #             "type": "image_url"
    #         },
    #         {
    #             "text": "图里有什么",
    #             "type": "text"
    #         }
    #     ]
    # }]
    
    try:
        response = chat_model.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
        print(f"VL模型修正响应: {response.content}")
        
        # 解析JSON响应（更健壮）
        try:
            corrections = _extract_and_parse_json_array(response.content)
            return corrections
        except Exception as pe:
            print(f"VL模型修正JSON解析失败: {pe}")
            return []
    except Exception as e:
        print(f"VL模型修正失败: {e}")
        return []

def handle_position(user_id:str, chat_id:str):
    chat_model = ChatModel(
        model_name=os.environ.get("VLM_MODEL_NAME", "doubao-seed-1-6-vision-250815"),
        api_key=os.environ.get("VLM_API_KEY", "dc7e10e7-1095-40ae-a172-3a7d16fc1e61"),
        api_base=os.environ.get("VLM_API_BASE", "https://ark.cn-beijing.volces.com/api/v3"),
        model_provider="openai"
    )
    embedding_model = EmbeddingModel(
        model_name=os.environ.get("EMB_MODEL_NAME", "doubao-embedding-large-text-250515"),
        api_key=os.environ.get("VLM_API_KEY", "dc7e10e7-1095-40ae-a172-3a7d16fc1e61"),
        api_base=os.environ.get("VLM_API_BASE", "https://ark.cn-beijing.volces.com/api/v3"),
    )
    # desc_vec = embedding_model.embed("海报")
    # user_info = UserInfoManager().get_user_info_by_user_id(user_id)
    # scene_id = user_info.current_scene_id
    # items = SceneItemsManager().search_items_by_description_vector(scene_id, desc_vec, top_k=10)
    # print(f"desc_vec: {items}")
    look_url = f"https://aura-view-eye.tos-cn-beijing.volces.com/assets/{user_id}/{chat_id}/view_data/look.jpg"
    cam_url = f"https://aura-view-eye.tos-cn-beijing.volces.com/assets/{user_id}/{chat_id}/view_data/cam.json"
    depth_url = f"https://aura-view-eye.tos-cn-beijing.volces.com/assets/{user_id}/{chat_id}/view_data/depth.exr"

    # 直接下载为内存数据（不落地临时文件）
    try:
        print(f"正在下载 look.jpg: {look_url}")
        resp_look = requests.get(look_url, timeout=30)
        resp_look.raise_for_status()
        look_bytes = np.frombuffer(resp_look.content, dtype=np.uint8)
        look_img = cv2.imdecode(look_bytes, cv2.IMREAD_COLOR)
        if look_img is None:
            raise RuntimeError("look.jpg 解析失败")
        print("look.jpg 下载并解析完成")

        print(f"正在下载 cam.json: {cam_url}")
        resp_cam = requests.get(cam_url, timeout=30)
        resp_cam.raise_for_status()
        cam_data = json.loads(resp_cam.content.decode('utf-8'))
        print("cam.json 下载并解析完成")

        print(f"正在下载 depth.exr: {depth_url}")
        resp_depth = requests.get(depth_url, timeout=30)
        resp_depth.raise_for_status()
        depth_content = resp_depth.content
        print("depth.exr 下载完成（内存）")
    except Exception as e:
        raise Exception(f"下载资源失败: {e}")
    
    yolo_model_dir = os.environ.get("YOLO_MODEL_PATH")
    if yolo_model_dir:
        model_path = f"{yolo_model_dir}/yoloe-11l-seg-pf.pt"
    else:
        model_path = "./yoloe_model/yoloe-11l-seg-pf.pt"
    model = YOLO(model_path)
    # 处理完成后清理临时目录（函数级别）
    # 注意：下载异常时已清理并抛出异常；正常流程在函数末尾清理
    results = model.predict(look_img)
    # 解析results[0]的内容
    result = results[0]
    # result.save(f"look_{user_id}_{chat_id}.jpg")
    print("=== YOLO检测结果解析 ===")
    print(f"检测到的对象数量: {len(result.boxes) if result.boxes is not None else 0}")
    
    # 使用EasyOCR识别文字
    print("\n=== OCR文字识别 ===")
    ocr_results = fast_ocr(look_img)
    print(f"识别到的原始文字数量: {len(ocr_results)}")
    
    # 对OCR结果进行聚类
    print("\n=== OCR结果聚类 ===")
    clustered_ocr = cluster_ocr_results(ocr_results, distance_threshold=50)
    print(f"聚类后的文字对象数量: {len(clustered_ocr)}")
    
    for i, ocr_item in enumerate(clustered_ocr):
        print(f"  文字对象 {i+1}: '{ocr_item['text']}' (置信度: {ocr_item['confidence']:.3f})")
        print(f"    位置: ({ocr_item['x1']:.1f}, {ocr_item['y1']:.1f}) 到 ({ocr_item['x2']:.1f}, {ocr_item['y2']:.1f})")
    
    # 解析检测框信息并准备VL修正
    detection_results = []
    if result.boxes is not None:
        print("\n=== 原始YOLO检测结果 ===")
        for i, box in enumerate(result.boxes):
            # 获取坐标
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            conf = box.conf[0].cpu().numpy()  # 置信度
            cls = int(box.cls[0].cpu().numpy())  # 类别ID
            
            class_name = result.names.get(cls, 'Unknown') if hasattr(result, 'names') and result.names else 'Unknown'
            class_name = class_name.replace(' ', '_')
            detection_info = {
                'index': i + 1,
                'x1': float(xyxy[0]),
                'y1': float(xyxy[1]),
                'x2': float(xyxy[2]),
                'y2': float(xyxy[3]),
                'conf': float(conf),
                'class_id': cls,
                'class_name': class_name,
                'type': 'object'  # 标记为物体类型
            }
            detection_results.append(detection_info)
            
            print(f"  对象 {i+1}:")
            print(f"    坐标: x1={xyxy[0]:.1f}, y1={xyxy[1]:.1f}, x2={xyxy[2]:.1f}, y2={xyxy[3]:.1f}")
            print(f"    置信度: {conf:.3f}")
            print(f"    类别ID: {cls}")
            print(f"    类别名称: {class_name}")
    
    # 将聚类后的OCR结果添加到检测结果中
    print("\n=== 合并聚类后的OCR文字检测结果 ===")
    for i, ocr_item in enumerate(clustered_ocr):
        ocr_detection = {
            'index': len(detection_results) + i + 1,
            'x1': ocr_item['x1'],
            'y1': ocr_item['y1'],
            'x2': ocr_item['x2'],
            'y2': ocr_item['y2'],
            'conf': ocr_item['confidence'],
            'class_id': -1,  # OCR文字使用特殊ID
            'class_name': 'text',
            'text_content': ocr_item['text'],  # 存储聚类后的文字内容
            'type': 'text'  # 标记为文字类型
        }
        detection_results.append(ocr_detection)
        print(f"  文字对象 {i+1}: '{ocr_item['text']}' (置信度: {ocr_item['confidence']:.3f})")
    
    print(f"\n=== 总检测结果统计 ===")
    print(f"YOLO检测对象数量: {len([d for d in detection_results if d['type'] == 'object'])}")
    print(f"OCR识别文字数量: {len([d for d in detection_results if d['type'] == 'text'])}")
    print(f"总检测数量: {len(detection_results)}")
    
    # 分离YOLO和OCR结果
    # yolo_results = [d for d in detection_results if d['type'] == 'object']
    # ocr_results = [d for d in detection_results if d['type'] == 'text']
    
    # 读取图像尺寸用于像素->NDC转换
    try:
        img_h, img_w = (look_img.shape[0], look_img.shape[1]) if look_img is not None else (1080, 1920)
    except Exception:
        img_h, img_w = (1080, 1920)
    inv_mvp = None
    depth_linear = None
    # near_plane = 0.1
    # far_plane = 100.0
    
    try:
        print(f"正在解析 cam.json（内存）")
        print(f"cam.json 下载成功，包含 {len(cam_data)} 个键")
        # 优先读取分离的 proj/view 矩阵
        proj = cam_data.get('projection_matrix')
        view_m = cam_data.get('view_matrix')
        # # 近远裁剪面（若提供）
        # if isinstance(cam_data.get('near'), (int, float)):
        #     near_plane = float(cam_data['near'])
        # if isinstance(cam_data.get('far'), (int, float)):
        #     far_plane = float(cam_data['far'])
        if isinstance(proj, list) and len(proj) == 16 and isinstance(view_m, list) and len(view_m) == 16:
            proj_matrix = np.array([proj[i*4:(i+1)*4] for i in range(4)], dtype=np.float64).transpose()
            view_matrix = np.array([view_m[i*4:(i+1)*4] for i in range(4)], dtype=np.float64).transpose()
            print("已解析分离的 proj/view 矩阵 (4x4)")
            try:
                inv_proj_matrix = np.linalg.inv(proj_matrix)
                inv_view_matrix = np.linalg.inv(view_matrix)
                camera_position = inv_view_matrix[:3, 3]
                cam_fwd_h = inv_view_matrix @ np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
                camera_forward = cam_fwd_h[:3]
                n = np.linalg.norm(camera_forward)
                if n > 0:
                    camera_forward = camera_forward / n
            except Exception as e:
                print(f"计算 inv_proj/inv_view 或相机参数失败: {e}")
    except requests.exceptions.RequestException as e:
        print(f"下载 cam.json 失败: {e}")
    except json.JSONDecodeError as e:
        print(f"解析 cam.json 失败: {e}")
    except Exception as e:
        print(f"处理 cam.json 时出错: {e}")
    # 读取 EXR 线性深度图（内存）
    try:
        print(f"正在解析 depth.exr（内存）")
        if OpenEXR is not None and Imath is not None:
            depth_linear = decode_exr_from_bytes_via_tempfile(depth_content)
        else:
            # 回退：使用 OpenCV（若环境启用了 EXR）
            depth_buf = np.frombuffer(depth_content, dtype=np.uint8)
            depth_img = cv2.imdecode(depth_buf, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise RuntimeError("depth.exr 解码失败（OpenEXR 不可用且 OpenCV 未启用 EXR）")
            if depth_img.ndim == 3 and depth_img.shape[2] >= 1:
                depth_linear = depth_img[:, :, 0].astype(np.float64)
            else:
                depth_linear = depth_img.astype(np.float64)
        print(f"depth.exr 解析成功，shape={depth_linear.shape}")
    except Exception as e:
        print(f"读取 depth.exr 失败: {e}")
    # 视口矩形
    view_rect = (0, 0, img_w, img_h)
    
    # 批量存储：两阶段
    # 阶段1：计算世界坐标并基于距离复用已有 Item，仅更新坐标等动态字段
    # 阶段2：对未匹配的新项使用 VLM 更正，再批量写入
    if detection_results:
        try:
            print("\n=== 第一阶段：匹配已有项并批量更新坐标 ===")
            db_manager = SceneItemsManager()
            
            # 获取场景ID（用于距离匹配与写库）
            user_info = UserInfoManager().get_user_info_by_user_id(user_id)
            scene_id = user_info.current_scene_id
            # 先为全部检测计算世界坐标
            matched_nodes = []
            unmatched_indices = []
            world_positions = []
            world_sizes = []
            base_names: List[str] = []
            for idx, detection in enumerate(detection_results):
                # 初始名称/描述（仅未匹配用于二阶段纠正前的占位）
                if detection['type'] == 'object':
                    base_name = detection.get('class_name', 'Unknown')
                    base_desc = detection.get('class_name', '无描述')
                else:
                    base_name = 'text'
                    base_desc = detection.get('text_content', 'text')
                
                # 计算像素中心点
                cx_px = (detection['x2'] - detection['x1']) / 2 + detection['x1']
                cy_px = (detection['y2'] - detection['y1']) / 2 + detection['y1']
                translation_vec = [0.0, 0.0, 0.0]
                world_size = 0.3
                if inv_proj_matrix is not None and inv_view_matrix is not None and depth_linear is not None and camera_position is not None and camera_forward is not None:
                    try:
                        u0 = int(np.clip(np.floor(cx_px), 0, img_w - 1))
                        v0 = int(np.clip(np.floor(cy_px), 0, img_h - 1))
                        depth_value = float(depth_linear[v0, u0])
                        world = world_position_from_depth(
                            np.array([cx_px, cy_px], dtype=np.float64),
                            view_rect,
                            inv_view_matrix,
                            inv_proj_matrix,
                            camera_position,
                            camera_forward,
                            depth_value,
                        )
                        world_size = screen_distance_to_world_distance(
                            np.array([detection['x2'], detection['y2']], dtype=np.float64),
                            np.array([detection['x1'], detection['y1']], dtype=np.float64),
                            depth_value,
                            depth_value,
                            view_rect,
                            inv_view_matrix,
                            inv_proj_matrix,
                            camera_position,
                            camera_forward,
                        )
                        translation_vec = [float(world[0]), float(world[1]), float(world[2])]
                    except Exception as e:
                        print(f"屏幕->世界坐标转换失败(idx={idx}): {e}")
                world_positions.append(translation_vec)
                world_sizes.append(world_size)
                base_names.append(base_name)
            try:
                if 'proj_matrix' in locals() and 'view_matrix' in locals():
                    view_proj_matrix = proj_matrix @ view_matrix
                    existing_items = db_manager.get_scene_items_by_distance(camera_position, 100000, scene_id, "MHC_Talker")
                    existing_items = db_manager.get_scene_items_in_frustum_by_view_proj(view_proj_matrix, existing_items)
            except Exception as _:
                pass
            item_type_groups = {}
            for it in existing_items:
                type = (it.get('item_type'))
                if type not in item_type_groups:
                    item_type_groups[type] = []
                item_type_groups[type].append(it)
            # AABB 相交匹配（同类型/前缀），相交则认为同一ID；如多项相交，取交叠体积最大的
            def nearest_existing_id(pos: list[float], base_name: str, size: float) -> str:
                half = max(0.01, float(size) / 2.0)
                new_min = [pos[0] - half, pos[1] - half, pos[2] - half]
                new_max = [pos[0] + half, pos[1] + half, pos[2] + half]

                def overlap1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
                    return max(0.0, min(a_max, b_max) - max(a_min, b_min))

                def overlap_volume(a_min, a_max, b_min, b_max) -> float:
                    ox = overlap1d(a_min[0], a_max[0], b_min[0], b_max[0])
                    oy = overlap1d(a_min[1], a_max[1], b_min[1], b_max[1])
                    oz = overlap1d(a_min[2], a_max[2], b_min[2], b_max[2])
                    return ox * oy * oz

                best_id = None
                best_vol = 0.0
                candidates = item_type_groups.get(base_name, [])
                for it in candidates:
                    ex_min = [
                        float(it.get('world_bb_x', it.get('world_pos_x', 0.0))),
                        float(it.get('world_bb_y', it.get('world_pos_y', 0.0))),
                        float(it.get('world_bb_z', it.get('world_pos_z', 0.0)))
                    ]
                    ex_max = [
                        ex_min[0] + float(it.get('world_bb_w', 0.0)),
                        ex_min[1] + float(it.get('world_bb_h', 0.0)),
                        ex_min[2] + float(it.get('world_bb_d', 0.0))
                    ]
                    vol = overlap_volume(new_min, new_max, ex_min, ex_max)
                    if vol > best_vol:
                        best_vol = vol
                        best_id = it.get('item_id')

                return best_id if best_vol > 0.0 else None
            for idx, detection in enumerate(detection_results):
                pos = world_positions[idx]
                size = world_sizes[idx]
                if pos is None:
                    unmatched_indices.append(idx)
                    continue
                same_id = nearest_existing_id(pos, base_names[idx], size)
                if same_id:
                    # 找到最近且足够近的已有条目：更新该 id 坐标
                    # 使用已有名称/描述（从 visible_existing 找）
                    found = next((it for it in existing_items if it.get('item_id') == same_id), None)
                    base_name = detection.get('class_name', 'Unknown') if detection['type'] == 'object' else 'text'
                    base_desc = detection.get('class_name', base_name) if detection['type'] == 'object' else detection.get('text_content','text')
                    half = max(0.01, float(size) / 2.0)
                    bb_min = [pos[0] - half, pos[1] - half, pos[2] - half]
                    bb_max = [pos[0] + half, pos[1] + half, pos[2] + half]
                    node_data = {
                        'item_type': (found or {}).get('item_type', 'Unknown'),
                        'item_name': (found or {}).get('item_name', 'Unknown'),
                        'description': (found or {}).get('description', base_desc),
                        'description_vector': [],
                        'translation': pos,
                        'extras': {
                            'tags': [(found or {}).get('item_id', base_name)],
                            'boundingBox': {
                                'min': bb_min,
                                'max': bb_max
                            },
                            'actions': {},
                            'skills': {},
                            'detection_type': detection['type'],
                            'confidence': detection['conf'],
                            'class_id': detection.get('class_id', -1),
                            'original_class': detection.get('class_name', 'Unknown')
                        }
                    }
                    matched_nodes.append(node_data)
                else:
                    unmatched_indices.append(idx)
            
            # 第一阶段：仅更新匹配到的项
            if matched_nodes:
                # 注意：upsert 会以 tags[0] 作为 item_name，item_id 将按名称去重生成；
                # 这里仅用于更新已有项的位置，因此 tags[0] 使用已有 item_name
                _ = db_manager.upsert_scene_items_batch(scene_id, matched_nodes)
            
            # 第二阶段：对未匹配的新项进行 VLM 更正后写入
            if unmatched_indices:
                print("\n=== 第二阶段：对未匹配项进行 VLM 更正并写入 ===")
                # 拆分对象与文字
                unmatched_objects = [detection_results[i] for i in unmatched_indices if detection_results[i]['type'] == 'object']
                unmatched_texts_raw = [detection_results[i] for i in unmatched_indices if detection_results[i]['type'] == 'text']
                # 调整文字为 VL 所需精简格式
                unmatched_texts = [
                    {
                        'index': k + 1,
                        'x1': t['x1'], 'y1': t['y1'], 'x2': t['x2'], 'y2': t['y2'],
                        'conf': t['conf'], 'text_content': t.get('text_content', '')
                    }
                    for k, t in enumerate(unmatched_texts_raw)
                ]
                # VLM 更正
                obj_corr = correct_detection_with_vl(chat_model, look_url, unmatched_objects) if unmatched_objects else []
                txt_corr = correct_ocr_with_vl(chat_model, look_url, unmatched_texts) if unmatched_texts else []
                # 组装第二批待写入节点（新项）
                batch_new_nodes = []
                # 对象
                for j, det in enumerate(unmatched_objects):
                    corrected_name = obj_corr[j].get('corrected_class', det.get('class_name', 'Unknown')) if obj_corr and j < len(obj_corr) else det.get('class_name', 'Unknown')
                    corrected_desc = obj_corr[j].get('desc', corrected_name) if obj_corr and j < len(obj_corr) else corrected_name
                    desc_vec = embedding_model.embed(corrected_desc) if corrected_desc else []
                    idx = unmatched_indices[[i for i, k in enumerate(unmatched_indices) if detection_results[k] is det][0]]
                    wp = world_positions[idx]
                    ws = world_sizes[idx] if idx < len(world_sizes) else 0.3
                    half = max(0.01, float(ws) / 2.0)
                    bb_min = [wp[0] - half, wp[1] - half, wp[2] - half]
                    bb_max = [wp[0] + half, wp[1] + half, wp[2] + half]
                    item_type = unmatched_objects[j].get('class_name', 'Unknown')
                    index = len(item_type_groups[item_type]) if item_type in item_type_groups else 0
                    item_id = f"{item_type}_{index}"
                    node = {
                        'item_type': item_type,
                        'item_name': corrected_name,
                        'description': corrected_desc,
                        'description_vector': desc_vec,
                        'translation': wp,
                        'extras': {
                            'tags': [item_id],
                            'boundingBox': {
                                'min': bb_min,
                                'max': bb_max
                            },
                            'actions': {},
                            'skills': {},
                            'detection_type': det['type'],
                            'confidence': det['conf'],
                            'class_id': det.get('class_id', -1),
                            'original_class': det.get('class_name', 'Unknown')
                        }
                    }
                    batch_new_nodes.append(node)
                # 文字
                for j, t in enumerate(unmatched_texts_raw):
                    text_val = txt_corr[j].get('text_content', t.get('text_content', '')) if txt_corr and j < len(txt_corr) else t.get('text_content', '')
                    desc_val = f"文字: {text_val}" if text_val else 'text'
                    desc_vec = embedding_model.embed(desc_val) if desc_val else []
                    idx = unmatched_indices[[i for i, k in enumerate(unmatched_indices) if detection_results[k] is t][0]]
                    wp = world_positions[idx]
                    ws = world_sizes[idx] if idx < len(world_sizes) else 0.3
                    half = max(0.01, float(ws) / 2.0)
                    bb_min = [wp[0] - half, wp[1] - half, wp[2] - half]
                    bb_max = [wp[0] + half, wp[1] + half, wp[2] + half]
                    index = len(item_type_groups['text']) if 'text' in item_type_groups else 0
                    item_id = f"text_{index}"
                    node = {
                        'item_type': 'text',
                        'item_name': '文字内容',
                        'description': desc_val,
                        'description_vector': desc_vec,
                        'translation': wp,
                        'extras': {
                            'tags': [item_id],
                            'boundingBox': {
                                'min': bb_min,
                                'max': bb_max
                            },
                            'actions': {},
                            'skills': {},
                            'detection_type': 'text',
                            'confidence': t['conf'],
                            'class_id': t.get('class_id', -1),
                            'original_class': 'text'
                        }
                    }
                    batch_new_nodes.append(node)
                if batch_new_nodes:
                    _ = db_manager.upsert_scene_items_batch(scene_id, batch_new_nodes)
                
        except Exception as e:
            print(f"❌ 批量存储对象到数据库时发生错误: {e}")
    # 无需清理临时目录（已全程使用内存数据）

