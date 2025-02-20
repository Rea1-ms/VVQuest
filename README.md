# VVQuest

VVQuest 是一个能够通过自然语言描述检索合适的张维为表情包的项目，运用嵌入模型检索。

[在线体验](https://vv.xy0v0.top)

## 数据来源

本项目张维为表情包来源于 [知乎](https://www.zhihu.com/question/656505859/answer/55843704436)

若有侵权，请联系删除

## 项目使用

1. git clone本仓库
2. 安装依赖
```bash
pip install -r requirements.txt
```
3. 获取API_KEY（可选，可使用本地模型）

    注册Silicon Flow账号后在[此处](https://cloud.siliconflow.cn/account/ak)获取

4. 运行项目
```bash
python -m streamlit run streamlit_app.py
```

## 添加额外图片
修改 `config/config.yaml` 中的 `paths.image_dirs`，添加示例配置如下：

```yaml
pic_example:
  path: 'path/to/your/images'
  regex:
    pattern: '^[^-]*-'
    replacement: ""
  type: "名称"
```

## Demo

<img width="256" alt="3bfb772e239f3437a13d46252aab1e1d" src="https://github.com/user-attachments/assets/d7e02f8f-205d-42ef-9c80-49f98aff64a6" />

## LICENSE


check [LICENSE](LICENSE)
