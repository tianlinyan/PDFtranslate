# AI 配置手册（简版）

翻译工具靠一个叫 `models.json` 的文件来认识“该用哪个 AI 模型”。程序启动后，你在主界面下拉框里能看到的所有模型，都来自这个文件。

普通用户**通常只需要改三个地方**：模型名（`model`）、服务器地址（`endpoint`）、密钥（`api_key`）。本手册只教这三个，够用。

---

## 一、models.json 在哪里

- **自己跑源码**：在项目根目录（和 `main.py` 放一起）。
- **打包版**：在 `.exe` 所在的那个文件夹里。

用**记事本**打开它。改完记得**保存**，然后**重启程序**才会生效。

---

## 二、一个本地模型的示例

本地模型（跑在自己电脑上，如 llama.cpp、Ollama）：

```json
{
  "id": "local-qwen",
  "name": "本地 Qwen",
  "endpoint": "http://127.0.0.1:8080/v1/chat/completions",
  "model": "qwen3.8"
}
```

---

## 三、一个云端模型的示例

云端模型（在线 API，如 DeepSeek、OpenAI）：

```json
{
  "id": "cloud-ds",
  "name": "DeepSeek",
  "endpoint": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-v4-flash",
  "api_key": "${DEEPSEEK_API_KEY}"
}
```

---

## 四、你只需要改这三处

| 你要填什么 | 这个字段 | 说明 |
|-----------|---------|------|
| **模型叫什么** | `model` | 填服务端认识的模型名。**云模型务必填对**，填错会说参数错误。 |
| **去哪个地址** | `endpoint` | 服务器的完整地址，**结尾必须是 `/chat/completions`**。本地一般是 `http://127.0.0.1:端口/...`，云端是厂商给的一串网址。 |
| **密码（密钥）** | `api_key` | 只有云端要填。本地模型一般不用填，可以直接删掉这一行。 |

> `id` 和 `name` 随便起，只是显示用。`id` 别乱改，改了会当作新模型、之前的翻译缓存就失效了。

---

## 五、密钥怎么填（重要）

**不要**把密钥明文写进 `models.json`。写成 `${变量名}` 这种占位符，密钥放环境变量里：

```json
"api_key": "${DEEPSEEK_API_KEY}"
```

然后在运行前设置好这个变量（Windows PowerShell）：

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
python main.py
```

> 放心：如果这个环境变量没设置，程序**不会**把 `${DEEPSEEK_API_KEY}` 当密钥发出去，只会提示你没配置，不会泄露。

---

## 六、常见问题（按提示处置）

- **“缺少 endpoint” / “缺少 model”**：上面两个字段没填，补上。
- **“api_key 环境变量未设置”**：你先设好了环境变量再用，或者本地模型把这行删了。
- **4xx 报错（认证/参数错）**：多半是 `api_key` 或 `model` 填得不符，对照厂商文档改。
- **翻译很慢**：云端可以给模型加 `"concurrency": 2`（最多到 4）提速；本地模型不建议，多半没用。
- **本地模型报 500**：尝试给模型加一行 `"reasoning_effort": "none"`，或改成 `low`。
- **出现“请求过多/429”**：把 `concurrency` 调回 1，或减小 `batch_size`（默认 4000，改成 2000）。

---

## 七、一个能直接用的完整例子

把下面这段整体放进 `models.json` 的 `"models": [ ... ]` 里，就是一个本地模型 + 一个云端模型：

```json
{
  "models": [
    {
      "id": "local-qwen",
      "name": "本地 Qwen",
      "endpoint": "http://127.0.0.1:8080/v1/chat/completions",
      "model": "qwen3.8"
    },
    {
      "id": "cloud-ds",
      "name": "DeepSeek",
      "endpoint": "https://api.deepseek.com/v1/chat/completions",
      "model": "deepseek-v4-flash",
      "api_key": "${DEEPSEEK_API_KEY}"
    }
  ]
}
```

改完保存 → 重启程序 → 下拉框选中它 → 翻译一个小文档试试。有问题就按第 6 节对照排查。
