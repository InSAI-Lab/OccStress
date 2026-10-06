from gtts import gTTS
from io import BytesIO
import pygame

# 初始化 Pygame 音频系统
pygame.mixer.init()

# 创建一个字节流作为临时音频存储
audio_buffer = BytesIO()

# 输入你想要转换为语音的文本
text = "Hello, how are you doing today?"

# 使用 gTTS 生成语音并保存到字节流
tts = gTTS(text, lang='en')
tts.write_to_fp(audio_buffer)

# 将字节流的指针重置到开头
audio_buffer.seek(0)

# 使用 Pygame 播放音频流
pygame.mixer.music.load(audio_buffer, 'mp3')
pygame.mixer.music.play()

# 保持程序在音频播放完毕之前不退出
while pygame.mixer.music.get_busy():
    continue
