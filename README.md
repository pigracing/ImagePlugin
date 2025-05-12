# ImagePlugin
图片处理插件，支持文生图，图生图，图生文的功能,适配xxxbot-pad版本的ImagePlugin，支持配置关键字进行不同API的调用，兼容OpenAI格式的API接口
安装后请注意修改config.toml的配置内容

# config.toml配置说明

```bash
[ImagePlugin.keywords."#query"]  #query为关键字
open_ai_api_url = "https://api.openai.com/v1"   API的基础URL
api-key =  "sk-cxxcxxxxxxx"   鉴权key
open_ai_model =  "gpt-4o"    调用模型
prompt = "你是一个优秀的图片生成专家，你负责根据客户的要求生成图片"   提示词
image_regex = "https://[^\\s\\)]+?\\.png"     提取图片的正则表达式
```

# 使用说明

1、发送图片给AI

2、在5分钟之内引用该图片，输入#query+需要让AI执行的内容，例如#query分析这张图片都有什么

3、后台AI返回结果，支持通过正则表达式提取图片，支持url返回及base64的格式返回

<div align="center">
<img width="700" src="./doc/1747026553701.jpg">
</div>
