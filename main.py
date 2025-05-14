import tomllib
import aiohttp
import asyncio
import base64
import mimetypes
import re
from typing import Any, Dict
from loguru import logger
from dataclasses import dataclass, field
import json
import time
from io import BytesIO
from PIL import Image

from WechatAPI import WechatAPIClient
from utils.decorators import *
from utils.plugin_base import PluginBase
import xml.etree.ElementTree as ET
import io
import traceback

@dataclass
class ModelConfig:
    open_ai_api_url: str
    api_key: str
    open_ai_model: str
    prompt: str
    image_regex: str
    is_translate: bool

class ImagePlugin(PluginBase):
    description = "图片处理插件，支持文生图，图生图，图生文的功能"
    author = "pigracing"
    version = "1.0.2"

    def __init__(self):
        super().__init__()

        with open("plugins/ImagePlugin/config.toml", "rb") as f:
            plugin_config = tomllib.load(f)

        config = plugin_config["ImagePlugin"]
        self.enable = config["enable"]
        # 加载所有模型配置
        self.keywords = config.get("keywords", {})
        logger.debug(f"加载模型配置1: {self.keywords}")
        for _keyword, _config in self.keywords.items():
            logger.debug(f"加载模型配置2: {_config}")
            self.keywords[_keyword] = ModelConfig(
                open_ai_api_url=_config["open_ai_api_url"],
                api_key=_config["api-key"],
                open_ai_model=_config["open_ai_model"],
                prompt=_config["prompt"],
                image_regex=_config.get("image_regex",""),
                is_translate=_config.get("is_translate",False)
            )
        logger.debug(f"加载模型配置3: {self.keywords}")
        self.image_cache = {}
    
    def match_keyword_name(self, text: str) -> str | None:
        for _keyword in self.keywords:
            if text.startswith(_keyword):
                return _keyword
        return None
    
    @on_xml_message(priority=100)  # 使用最高优先级确保最先处理
    async def handle_xml_quote(self, bot: WechatAPIClient, message: dict):
        """专门处理XML格式的引用消息"""
        logger.debug("ImagePlugin---on_xml_quote_message")
        #logger.debug(message)
        if not self.enable:
            logger.debug("ImagePlugin---on_xml_quote_message--not enable")
            return True
        else:
            if message["Quote"] is None:
                logger.debug("ImagePlugin---on_xml_quote_message--Quote is None")
                return True
            else:
                if message["Quote"]["MsgType"] == 3:
                    logger.debug("ImagePlugin---on_xml_quote_message--MsgType is 3")
                    return False
        return True

    @on_text_message
    async def handle_text(self, bot: WechatAPIClient, message: dict):
        logger.debug("ImagePlugin---on_quote_message")
        if not self.enable:
            return

        matched_name = self.match_keyword_name(message["Content"])
        if matched_name:
            logger.debug(f"匹配到关键字: {matched_name}")
        else:
            logger.debug("没有匹配到关键字")
            return
        
        content = message["Content"]
        content = content[len(matched_name):].strip()
        logger.debug("处理内容: " + content)
        try:
            out_message = await self.call_openai_api(self.keywords[matched_name], [{"role": "user", "content": content}])
            logger.debug("返回内容: " + out_message)
            if self.is_image_url(out_message):
                # 如果返回的是图片链接，直接发送
                base64_str = await self.image_url_to_base64(out_message)
                logger.debug("图片链接转换为base64: " + base64_str[:100])
                await bot.send_image_message(message["FromWxid"], base64_str)
            else:  
                # 如果返回的是文本，直接发送
                await bot.send_text_message(message["FromWxid"], out_message)
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await bot.send_text_message(message["FromWxid"], "处理消息失败，请稍后再试。")
        return False  # 阻止后续插件处理
    
    @on_quote_message(priority=30)
    async def handle_text(self, bot: WechatAPIClient, message: dict):
        logger.debug("ImagePlugin---on_quote_message")
        if not self.enable:
            return True
        logger.debug(f"ImagePlugin---{message}")
        newMsgId = message["Quote"]["NewMsgId"]
        logger.debug("ImagePlugin---on_quote_message------"+newMsgId)
        logger.debug(len(self.image_cache))
        try:
            img_data_base64 = self.image_cache[newMsgId]
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        content = message["Content"]
        logger.debug("ImagePlugin---on_quote_message------"+newMsgId+","+img_data_base64[:100])
        matched_name = self.match_keyword_name(content)
        if matched_name:
            logger.debug(f"ImagePlugin匹配到关键字: {matched_name}")
        else:
            logger.debug("ImagePlugin没有匹配到关键字,请检查配置")
            return True
        
        content = content[len(matched_name):].strip()
        logger.debug("ImagePlugin用户指令内容: " + content)
        try:
            _config = self.keywords[matched_name]
            if img_data_base64 is None:
                logger.error("ImagePlugin 未找到图片，请重新尝试")
                return False
            img_data = f"data:image/png;base64,{img_data_base64}"
            user_text = [
                {
                    "type": "text",
                    "text": content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":img_data
                    }
                }
            ]
            _messages = [{"role": "system", "content": _config.prompt},{"role": "user", "content": user_text}]
            out_message = await self.call_openai_api(_config, _messages)
            logger.debug("返回内容: " + out_message)
            if self.is_image_url(out_message):
                # 如果返回的是图片链接，直接发送
                base64_str = await self.image_url_to_base64(out_message)
                logger.debug("图片链接转换为base64: " + base64_str[:100])
                await bot.send_image_message(message["FromWxid"], base64_str)
            elif self.is_base64_image(out_message):
                if out_message.startswith("data:image"):
                    out_message = out_message.split(',', 1)[1]
                logger.debug("图片返回的内容为base64: " + out_message[:100])
                await bot.send_image_message(message["FromWxid"], out_message)
            else:  
                # 如果返回的是文本，直接发送
                await bot.send_text_message(message["FromWxid"], out_message)
            return False  # 阻止后续插件处理
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            # await bot.send_text_message(message["FromWxid"], "处理消息失败，请稍后再试。")
            return False  # 阻止后续插件处理

    
    @on_image_message(priority=30)
    async def handle_image(self, bot: WechatAPIClient, message: dict):
        """处理图片消息"""
        if not self.enable:
            return True
        try:
            logger.debug("ImagePlugin-----处理图片消息")
            # message["Content"] = message["Content"][:100]
            # logger.debug(message)
            # 获取图片消息的关键信息
            msg_id = message.get("MsgId")
            from_wxid = message.get("FromWxid")
            sender_wxid = message.get("SenderWxid")
            newMsgId = message.get("NewMsgId")
            isGroup = message.get("IsGroup")
            logger.info(f"收到图片消息: MsgId={msg_id}, FromWxid={from_wxid}, SenderWxid={sender_wxid},newMsgId={newMsgId}, IsGroup={isGroup}")
            if isGroup:
                # 尝试使用新的get_msg_image方法分段下载图片
                try:
                    length = message.get("ImageInfo").get("length")
                    aeskey = message.get("ImageInfo").get("aeskey")
                    cdnmidimgurl = message.get("ImageInfo").get("cdnmidimgurl")
                    logger.info(f"收到图片消息: length={length}, aeskey={aeskey}, cdnmidimgurl={cdnmidimgurl}")
                    if length and length.isdigit():
                        img_length = int(length)
                        logger.debug(f"尝试使用get_msg_image下载图片: MsgId={message.get('MsgId')}, length={img_length}")

                        # 分段下载图片
                        chunk_size = 64 * 1024  # 64KB
                        chunks = (img_length + chunk_size - 1) // chunk_size  # 向上取整
                        full_image_data = bytearray()

                        logger.info(f"开始分段下载图片，总大小: {img_length} 字节，分 {chunks} 段下载")

                        download_success = True
                        for i in range(chunks):
                            try:
                                # 下载当前段
                                start_pos = i * chunk_size
                                chunk_data = await bot.get_msg_image(message.get('MsgId'), message["FromWxid"], img_length, start_pos=start_pos)
                                if chunk_data and len(chunk_data) > 0:
                                    full_image_data.extend(chunk_data)
                                    logger.debug(f"第 {i+1}/{chunks} 段下载成功，大小: {len(chunk_data)} 字节")
                                else:
                                    logger.error(f"第 {i+1}/{chunks} 段下载失败，数据为空")
                                    download_success = False
                                    break
                            except Exception as e:
                                logger.error(f"下载第 {i+1}/{chunks} 段时出错: {e}")
                                download_success = False
                                break

                        if download_success and len(full_image_data) > 0:
                            # 验证图片数据
                            try:
                                import base64
                                from PIL import Image, ImageFile
                                ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图片

                                image_data = bytes(full_image_data)
                                # 验证图片数据
                                Image.open(io.BytesIO(image_data))
                                img_data_base64 = base64.b64encode(image_data).decode('utf-8')
                                logger.info(f"分段下载图片成功，总大小: {len(image_data)} 字节")
                            except Exception as img_error:
                                logger.error(f"验证分段下载的图片数据失败: {img_error}")
                                # 如果验证失败，尝试使用download_image
                                if aeskey and cdnmidimgurl:
                                    logger.warning("尝试使用download_image下载图片")
                                    img_data_base64 = await bot.download_image(aeskey, cdnmidimgurl)
                        else:
                            logger.warning(f"分段下载图片失败，已下载: {len(full_image_data)}/{img_length} 字节")
                            # 如果分段下载失败，尝试使用download_image
                            if aeskey and cdnmidimgurl:
                                logger.warning("尝试使用download_image下载图片")
                                img_data_base64 = await bot.download_image(aeskey, cdnmidimgurl)
                    elif aeskey and cdnmidimgurl:
                        logger.debug("使用download_image下载图片")
                        img_data_base64 = await bot.download_image(aeskey, cdnmidimgurl)
                except Exception as e:
                    logger.error(f"下载图片失败: {e}")
                    if aeskey and cdnmidimgurl:
                        try:
                            img_data_base64 = await bot.download_image(aeskey, cdnmidimgurl)
                        except Exception as e2:
                            logger.error(f"备用方法下载图片也失败: {e2}")
            else:
                img_data_base64 = message["ImgBuf"]["buffer"]
            logger.debug(img_data_base64[:100])
            self.image_cache[str(newMsgId)] = img_data_base64
            logger.debug("cache:"+str(newMsgId)+","+img_data_base64[:100])
            return False
        except Exception as e:
            logger.error(f"处理图片消息失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False


    async def call_openai_api(self,config: ModelConfig, messages: list[Dict[str, Any]]) -> str:
        url = f"{config.open_ai_api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.open_ai_model,
            "stream": False,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()

                    if response.status != 200:
                        raise RuntimeError(f"OpenAI API 请求失败: {response.status} - {response_text}")

                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"响应无法解析为 JSON（Content-Type: {response.headers.get('Content-Type')}）：\n{response_text}")

                    # 解析内容
                    text = data["choices"][0]["message"]["content"]

                    # 使用正则提取（如配置了 image_regex）
                    if config.image_regex:
                        matches = re.findall(config.image_regex, text)
                        if matches:
                            return "\n".join(matches)

                    return text
        except Exception as e:
            logger.error(f"调用 OpenAI API 失败: {e}")
            return False

    def is_image_url(self,url: str) -> bool:
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff')
        return url.lower().startswith("http") and url.lower().endswith(image_extensions)
    
    async def image_url_to_base64(self,url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise Exception(f"图片下载失败: HTTP {resp.status}")
                    content = await resp.read()
                    base64_str = base64.b64encode(content).decode("utf-8")
                    return base64_str

        except Exception as e:
            return f"[图片处理错误] {str(e)}"
        
    def is_base64_image(self,data: str) -> bool:
        try:
            # 如果有 data:image 开头，分离头部和 base64 部分
            if data.startswith("data:image"):
                header, data = data.split(",", 1)

            # 尝试解码 base64 字符串
            decoded_data = base64.b64decode(data, validate=True)

            # 尝试用 PIL 加载图片
            Image.open(BytesIO(decoded_data)).verify()
            return True
        except Exception:
            return False
