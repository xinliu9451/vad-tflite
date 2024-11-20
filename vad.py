import gradio as gr

import numpy as np
import tensorflow as tf
import librosa


class VADInference:
    def __init__(self, model_path, duration):
        # 模型参数
        self.SAMPLE_RATE = 16000
        self.DURATION = duration
        self.N_MELS = 40
        self.HOP_LENGTH = 512
        self.N_FFT = 2048

        # 加载TFLite模型
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # 获取输入和输出张量的细节
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_audio(self, audio_path, start=None, end=None):
        """预处理音频文件"""
        # 加载音频
        audio, rate = librosa.load(audio_path, sr=self.SAMPLE_RATE) 

        # 确保音频长度一致
        if len(audio) < self.SAMPLE_RATE * self.DURATION:
            audio = np.pad(audio, (0, int(self.SAMPLE_RATE * self.DURATION) - len(audio)))
        else:
            audio = audio[int(rate*start):int(rate*end)]

        # 提取梅尔频谱特征
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH
        )

        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 归一化
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()

        # 添加batch和channel维度
        mel_spec_db = np.expand_dims(mel_spec_db, axis=[0, -1])
        # 确保数据类型匹配模型输入
        mel_spec_db = mel_spec_db.astype(np.float32)

        return mel_spec_db, audio, rate

    def predict(self, audio_path, threshold=0.5, start=None, end=None):
        """对音频文件进行VAD预测"""
        # 预处理音频
        input_data, audio, rate = self.preprocess_audio(audio_path, start=start, end=end)

        # 设置输入张量
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # 运行推理
        self.interpreter.invoke()

        # 获取输出结果
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # 获取预测概率
        probability = output_data[0][0]

        # 根据阈值判断是否有人声
        has_voice = probability > threshold  # , np.expand_dims(audio, axis=0)

        return (rate, audio), {
            'has_voice': bool(has_voice),
            'probability': round(float(probability), 4)
        }
        

html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Introduce</h2>
    <p style="font-size: 18px;margin-left: 20px;">这是一个音频说话人识别的Demo。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Configuration Introduction</h2>
    <p style="font-size: 18px;margin-left: 20px;">Audio Duration：选择待识别音频的时长，目前只支持1秒钟的音频。</p>
    <p style="font-size: 18px;margin-left: 20px;">thresholds：设置阈值，音频识别的结果是一个概率分数，需设置一个阈值进行比较。</p>
    <p style="font-size: 18px;margin-left: 20px;">Starts：设置需要检测的音频的开始时间点，通过设置不同的时间点可以检测同一音频的不同片段。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2>
    <p style="font-size: 18px;margin-left: 20px;">上传音频格式为mp3，采样率16000，单声道。</p>
</div>
"""
audio_examples = [
["example_data/前1m_3.wav"],
["example_data/TEST_MEETING_T0000000000.mp3"],
["example_data/TEST_MEETING_T0000000001.mp3"],
["example_data/TEST_MEETING_T0000000002.mp3"],
["example_data/TEST_MEETING_T0000000003.mp3"],
["example_data/TEST_MEETING_T0000000004.mp3"],
["example_data/TEST_MEETING_T0000000005.mp3"],
["example_data/TEST_MEETING_T0000000006.mp3"],
["example_data/TEST_MEETING_T0000000007.mp3"],
["example_data/TEST_MEETING_T0000000008.mp3"],
["example_data/TEST_MEETING_T0000000009.mp3"],
["example_data/TEST_MEETING_T0000000010.mp3"],
]

def model_inference(audio_input, duration, thresholds, start):
    end = start + duration
    vad = VADInference(model_path=f"model/vad_model_1.0s.tflite", duration=duration)
    audio, result = vad.predict(audio_path=audio_input, threshold=thresholds, start=start, end=end)
    
    return audio, result

def launch():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(html_content)
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Configuration"):
                    duration = gr.Dropdown(choices=[1], value=1, label="Audio Duration")
                    thresholds = gr.Slider(minimum=0, maximum=1, step=0.01, label="thresholds", value=0.5)
                    start = gr.Number(label="Starts (Default:0)", value=0,  step=0.1, precision=1 )
                with gr.Row():
                    audio_input = gr.Audio(type="filepath", label="Upload Audio")

                fn_button = gr.Button("Start", variant="primary")
                audio_outputs = gr.Audio(label="Detected Audio")
                text_outputs = gr.Textbox(label="Detection Results")


            gr.Examples(examples=audio_examples, inputs=[audio_input], examples_per_page=20)

        fn_button.click(model_inference, inputs=[audio_input, duration, thresholds, start], outputs=[audio_outputs, text_outputs])

    demo.launch(share=True)


if __name__ == "__main__":
    launch()
