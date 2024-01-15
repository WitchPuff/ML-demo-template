import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Canvas, Scrollbar, Frame
import speech_recognition as sr
from PIL import Image, ImageTk
from utils import predict
from models import AlexNet, DenseNet
from utils import speech_to_text_baidu

window_size = (750, 700)
# 假设的模型预测API
def predict_model(image_path, model_type, dataset=r'data\kvasir-dataset-v2'):
    model_type = model_type.lower()
    if model_type == 'alexnet':
        model = AlexNet()
    elif model_type == 'densenet':
        model = DenseNet()
    label = predict(model, image_path, dataset)
    true_label = image_path.split('/')[-2]
    info = f"{model.model}模型预测结果:\n 图像路径 - {image_path}\n预测标签 - {label}\n真实标签 - {true_label}"
    return label, info

# 语音识别函数
def recognize_speech():
    # recognizer = sr.Recognizer()
    # with sr.Microphone() as source:
    #     print("请说话...")
    #     audio = recognizer.listen(source, timeout=2)

    # try:
        # 从麦克风读入
        print("请说话...")
        text = speech_to_text_baidu(if_microphone=True)
        # text = recognizer.recognize_google(audio, language='en-US', show_all=False)
        if "一" or "1" in text:
            return "AlexNet"
        elif "二" or '2' in text:
            return "DenseNet"
        else:
            return "Unrecognized"
    # except sr.UnknownValueError:
    #     return "无法识别语音"
    # except sr.RequestError:
    #     return "无法请求结果"
    # except sr.WaitTimeoutError:
    #     return "等待超时，请在开始录音后一秒内说话"

def resize_image(image, window_size):
    w, h = window_size
    return image.resize((w, h//3))

class App:
    def __init__(self, root):
        self.root = root
        self.root.geometry(f'{window_size[0]}x{window_size[1]}')
        # 创建滚动条
        self.canvas = Canvas(root)
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)
        self.scrollable_frame.config(width=window_size[0], height=window_size[1])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        self.create_start_page()
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def clear_widgets(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def create_start_page(self):
        self.clear_widgets()
        self.image1 = Image.open(r"models\AlexNet_training_history.png")
        self.image1 = resize_image(self.image1, window_size)
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.label1 = tk.Label(self.scrollable_frame, image=self.photo1)
        self.label1.image = self.photo1
        self.label1.pack(padx=5, pady=5)

        self.image2 = Image.open(r"models\DenseNet_training_history.png")
        self.image2 = resize_image(self.image2, window_size)
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.label2 = tk.Label(self.scrollable_frame, image=self.photo2)
        self.label2.image = self.photo2
        self.label2.pack(padx=5, pady=5)

        self.create_table()

        self.predict_button = tk.Button(self.scrollable_frame, text="预测", command=self.create_predict_page)
        self.predict_button.pack(pady=5)
        
        
    def create_table(self):
        columns = ("Model", "Best Train Acc", "Best Test Acc", "Time/Epoch(s)")
        self.table = ttk.Treeview(self.scrollable_frame, columns=columns, show='headings', height=2)
        # 设置列名
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=170, anchor='center')
        # 插入一行，左侧列显示 "AlexNet" 与 "DenseNet"，其余列初始化为 0
        self.table.insert("", "end", values=("AlexNet (batch_size=32)", 0.928, 0.85, 70))
        self.table.insert("", "end", values=("DenseNet (batch_size=16)", 0.958, 0.86, 401.87))
        
        self.table.pack(pady=5)
        text = "最优参数：SGD, learning_rate=1e-3, weight_decay=1e-4, momentum=0.9\n采用Early Stop Policy与学习率梯度下降策略，\n每5个epoch，valid loss未下降，则学习率减半，\n当learning_rate<=1e-5, early stop。"
        self.param_info = tk.Label(self.scrollable_frame, text="")
        self.param_info.config(text=text)
        self.param_info.pack()

    def create_predict_page(self):
        self.clear_widgets()
        self.upload_button = tk.Button(self.scrollable_frame, text="上传图像", command=self.upload_image)
        self.upload_button.pack(padx=window_size[0]//2, pady=5)

        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.scrollable_frame, textvariable=self.model_var)
        self.model_menu['values'] = ('AlexNet', 'DenseNet')
        self.model_menu.pack(pady=5)

        self.speech_button = tk.Button(self.scrollable_frame, text="语音选择", command=self.speech_recognition)
        self.speech_button.pack(pady=5)

        self.model_label = tk.Label(self.scrollable_frame, text="")
        self.model_label.pack(pady=0)
        
        self.confirm_button = tk.Button(self.scrollable_frame, text="确定", command=self.predict)
        self.confirm_button.pack(pady=5)

        self.return_button = tk.Button(self.scrollable_frame, text="返回", command=self.create_start_page)
        self.return_button.pack(pady=5)

        self.prediction_label = tk.Label(self.scrollable_frame, text="")
        self.prediction_label.pack(pady=5)

    def upload_image(self):
        new_image_path = filedialog.askopenfilename()
        if new_image_path:  # 确保用户选中了文件
            self.image_path = new_image_path
            self.image = Image.open(self.image_path)
            self.image_y = self.image_path.split('/')[-2]
            # 调整图像大小
            self.image = self.image.resize((400, 300))

            # 创建或更新图像标签
            if hasattr(self, 'image_label'):
                # 如果已经存在图像标签，先移除旧的标签
                self.image_label.destroy()
                self.prediction_label.config(text="")

            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label = tk.Label(self.scrollable_frame, image=self.photo)
            self.image_label.image = self.photo  # 引用保持，防止垃圾回收
            self.image_label.pack(pady=5)

    def speech_recognition(self):
        messagebox.showinfo("语音识别", '点击确定后开始录音')
        result = recognize_speech()
        if result in ["AlexNet", "DenseNet"]:
            self.model_var.set(result)
            messagebox.showinfo("语音识别", result)
        else:
            messagebox.showinfo("语音识别", '未提供该模型，\n请选择：AlexNet / DenseNet')

    def predict(self):
        self.model_label.config(text='预测中...')
        if not hasattr(self, 'photo'):
            messagebox.showwarning("警告", "请先上传图像")
        elif not self.model_var.get().lower() in ['alexnet', 'densenet']:
            messagebox.showwarning("警告", "请先选择提供的模型")
        else:
            label, info = predict_model(self.image_path, self.model_var.get())
            messagebox.showinfo("预测结果", f'Predicted: {label}\nTrue: {self.image_y}')
            self.prediction_label.config(text=info)
            self.model_label.config(text='')
            

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
