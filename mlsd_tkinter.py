import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import os
from PIL import Image, ImageTk
import sys
import test
from pathlib import Path
import scipy.io
import cv2

class Application(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master.title("VP with M-LSD")
        self.master.minsize(800, 660)
        self.master.maxsize(800, 660)
        self.master.geometry("800x660")

        # フレームの作成
        self.frame1 = tk.Frame(self.master, width=385, height=350, relief="ridge", bg="#a9a9a9")
        self.frame2 = tk.Frame(self.master, width=385, height=500, relief="ridge", bg="#a9a9a9")
        self.frame3 = tk.Frame(self.master, width=385, height=230, relief="ridge", bg="#a9a9a9")
        self.frame4 = tk.Frame(self.master, width=385, height=80, relief="ridge", bg="#a9a9a9")
        self.frame5 = tk.Frame(self.master, width=780, height=40, relief="ridge", bg="#a9a9a9")
        self.frame6 = tk.Frame(self.frame1, width=375, height=80, relief="ridge", bg="#808080")
        self.frame7 = tk.Frame(self.frame3, width=375, height=35, relief="ridge", bg="#808080")

        # VP用フレームラベル
        self.frame_vp=tk.LabelFrame(self.frame2, width=375, height=145,
                                    text="Vanishing Point",
                                    bg="#a9a9a9",
                                    font=("Lucida Console", "10", "bold"),
                                    bd=3)

        self.frame1.propagate(False)
        self.frame2.propagate(False)
        self.frame3.grid_propagate(False)
        self.frame4.propagate(False)
        self.frame5.propagate(False)
        self.frame6.propagate(False)
        self.frame7.propagate(False)
        self.frame_vp.grid_propagate(False)

        # ウィジェットの作成
        self.create_widgets()

        self.frame1.place(x=10, y=10)
        self.frame2.place(x=405, y=10)
        self.frame3.place(x=10, y=370)
        self.frame4.place(x=405, y=520)
        self.frame5.place(x=10, y=610)
        self.frame6.place(x=5, y=5)
        self.frame7.place(x=5, y=190)
        self.frame_vp.place(x=5,y=350)

    def create_widgets(self):
        self.btn_quit=tk.Button(self.frame5, text="QUIT", command=quit, bg="#a9a9a9", font=("Lucida Console", "20", "bold"), anchor=tk.CENTER)
        self.btn_quit.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # 左上の下 画像のファイルパス表示
        self.file_name=tk.StringVar()
        self.file_name.set("")
        self.img_name=tk.StringVar()
        self.img_name.set("Not selected")
        label = tk.Label(self.frame6, textvariable=self.img_name, font=("Lucida Console", "13", "bold"))
        label.pack(fill=tk.BOTH, side=tk.BOTTOM, padx=5, pady=5)

        # 左上の左 画像選択ボタン
        self.btn_input=tk.Button(self.frame6, text="select image", bg="#a9a9a9", font=("Lucida Console", "13", "bold"))
        self.btn_input.bind("<ButtonPress>", self.file_dialog)
        self.btn_input.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5)

        # 左上の右 画像取り消しボタン
        self.select_db=ttk.Combobox(self.frame6, state='readonly', font=("Lucida Console", "15", "bold"))
        self.select_db["values"]=("YorkUrbanDB", "Wireframe")
        self.select_db.current(0)
        self.select_db.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5)

        # self.btn_clear=tk.Button(self.frame6, text="select DB", bg="#808080", font=("Lucida Console", "13", "bold"))
        # self.btn_clear.bind("<ButtonPress>",self.clear_img)
        # self.btn_clear.pack(fill=tk.X, side=tk.RIGHT, expand=True, padx=5)

        # キャンバス(input)の初期化
        self.img_input = None
        self.canvas_input=tk.Canvas(self.frame1)
        self.canvas_input.place(x=5, y=90, width=375, height=255)
        self.update()
        self.canvas_input_width=self.canvas_input.winfo_width()
        self.canvas_input_height=self.canvas_input.winfo_height()
        self.canvas_img_input=self.canvas_input.create_image(self.canvas_input_width/2, self.canvas_input_height/2) # 画像の描画

        # キャンバス(output)の初期化
        self.img_output = None
        self.canvas_output=tk.Canvas(self.frame2)
        self.canvas_output.place(x=5, y=5, width=375, height=340)
        self.update()
        self.canvas_output_width=self.canvas_output.winfo_width()
        self.canvas_output_height=self.canvas_output.winfo_height()
        self.canvas_img_output=self.canvas_output.create_image(self.canvas_output_width/2, self.canvas_output_height/2) # 画像の描画

        # スライダーの初期化
        self.scale_var_dist = tk.DoubleVar()
        self.scale_dist_thr = tk.Scale(self.frame3,
                    variable = self.scale_var_dist,
                    command = self.slider_scroll_dist,
                    orient=tk.HORIZONTAL,   # 配置の向き、水平(HORIZONTAL)、垂直(VERTICAL)
                    length = 272,           # 全体の長さ
                    width = 7,             # 全体の太さ
                    sliderlength = 15,      # スライダー（つまみ）の幅
                    from_ = 0,            # 最小値（開始の値）
                    to = 20,               # 最大値（終了の値）
                    resolution=1,         # 変化の分解能(初期値:1)
                    tickinterval=4,         # 目盛りの分解能(初期値0で表示なし)
                    bg="#a9a9a9",
                    bd=1
                    )
        self.scale_var_score = tk.DoubleVar()
        self.scale_score_thr = tk.Scale(self.frame3,
                    variable = self.scale_var_score,
                    command = self.slider_scroll_score,
                    orient=tk.HORIZONTAL,   # 配置の向き、水平(HORIZONTAL)、垂直(VERTICAL)
                    length = 272,           # 全体の長さ
                    width = 7,             # 全体の太さ
                    sliderlength = 15,      # スライダー（つまみ）の幅
                    from_ = 0,            # 最小値（開始の値）
                    to = 1,               # 最大値（終了の値）
                    resolution=0.1,         # 変化の分解能(初期値:1)
                    tickinterval=0.2,         # 目盛りの分解能(初期値0で表示なし)
                    bg="#a9a9a9",
                    bd=1
                    )
        self.scale_var_len = tk.DoubleVar()
        self.scale_len_thr = tk.Scale(self.frame3,
                    variable = self.scale_var_len,
                    command = self.slider_scroll_len,
                    orient=tk.HORIZONTAL,   # 配置の向き、水平(HORIZONTAL)、垂直(VERTICAL)
                    length = 272,           # 全体の長さ
                    width = 7,             # 全体の太さ
                    sliderlength = 15,      # スライダー（つまみ）の幅
                    from_ = 1,            # 最小値（開始の値）
                    to = 60,               # 最大値（終了の値）
                    resolution=1,         # 変化の分解能(初期値:1)
                    tickinterval=11,        # 目盛りの分解能(初期値0で表示なし)
                    bg="#a9a9a9",
                    bd=1
                    )
        self.scale_dist_thr.set(10)
        self.scale_score_thr.set(0.2)
        self.scale_len_thr.set(10)


        # スライダーのラベル作成
        label_score = tk.Label(self.frame3, text="score  ->", font=("Lucida Console", "10", "bold"),bg="#a9a9a9")
        label_dist = tk.Label(self.frame3, text="dist   ->", font=("Lucida Console", "10", "bold"), bg="#a9a9a9")
        label_len = tk.Label(self.frame3, text="length ->", font=("Lucida Console", "10", "bold"), bg="#a9a9a9")
        label_score.grid(row=0,column=0, padx=5, pady=5)
        self.scale_score_thr.grid(row=0,column=1, padx=5, pady=5, sticky=tk.E)
        label_dist.grid(row=1,column=0, padx=5, pady=5)
        self.scale_dist_thr.grid(row=1,column=1, padx=5, pady=5, sticky=tk.E)
        label_len.grid(row=2,column=0, padx=5, pady=5)
        self.scale_len_thr.grid(row=2,column=1, padx=5, pady=5, sticky=tk.E)

        # スライダー初期化ボタン
        self.slider_initial=tk.Button(self.frame7, text="reset value", bg="#a9a9a9", font=("Lucida Console", "11", "bold"))
        self.slider_initial.bind("<ButtonPress>", self.reset)
        self.slider_initial.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5, pady=5)
        # 実行ボタン
        self.btn_run=tk.Button(self.frame7, text="run M-LSD", bg="#a9a9a9", font=("Lucida Console", "20", "bold"))
        self.btn_run.bind("<ButtonPress>",self.run)
        self.btn_run.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5, pady=5)

        # 右下の上 結果保存ボタン
        self.btn_save=tk.Button(self.frame4, text="SAVE Image", bg="#a9a9a9", font=("Lucida Console", "16", "bold"))
        self.btn_save.bind("<ButtonPress>", self.save_image)
        self.btn_save.pack(fill=tk.BOTH, side=tk.TOP, padx=5, pady=5)

        # 右下の左 結果クリアボタン
        self.btn_result_clear=tk.Button(self.frame4, text="clear result", bg="#a9a9a9", font=("Lucida Console", "13", "bold"))
        self.btn_result_clear.bind("<ButtonPress>", self.clear_result)
        self.btn_result_clear.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5, pady=5)

        # 右下の右 全部初期化ボタン
        self.btn_all_clear=tk.Button(self.frame4, text="ALL clear", bg="#a9a9a9", font=("Lucida Console", "13", "bold"))
        self.btn_all_clear.bind("<ButtonPress>",self.all_clear)
        self.btn_all_clear.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5, pady=5)

        # vpのラベル作成
        label_GT = tk.Label(self.frame_vp, text="   GT   ", font=("Lucida Console", "15", "bold"), bg="#a9a9a9")
        label_mlsd = tk.Label(self.frame_vp, text="  MLSD  ", font=("Lucida Console", "15", "bold"), bg="#a9a9a9")
        label_vp1 = tk.Label(self.frame_vp, text="VP1", font=("Lucida Console", "11", "bold"), bg="#a9a9a9")
        label_vp2 = tk.Label(self.frame_vp, text="VP2", font=("Lucida Console", "11", "bold"), bg="#a9a9a9")
        label_vp3 = tk.Label(self.frame_vp, text="VP3", font=("Lucida Console", "11", "bold"), bg="#a9a9a9")

        self.gt_value1=tk.StringVar()
        self.gt_value1.set("0,0")
        self.gt_value2=tk.StringVar()
        self.gt_value2.set("0,0")
        self.gt_value3=tk.StringVar()
        self.gt_value3.set("0,0")
        self.mlsd_value1=tk.StringVar()
        self.mlsd_value1.set("0,0")
        self.mlsd_value2=tk.StringVar()
        self.mlsd_value2.set("0,0")
        self.mlsd_value3=tk.StringVar()
        self.mlsd_value3.set("0,0")
        label_GT_value1 = tk.Label(self.frame_vp, textvariable=self.gt_value1, font=("Lucida Console", "10", "bold"), width=17, bg="#a9a9a9")
        label_GT_value2 = tk.Label(self.frame_vp, textvariable=self.gt_value2,font=("Lucida Console", "10", "bold"), width=17 ,bg="#a9a9a9")
        label_GT_value3 = tk.Label(self.frame_vp, textvariable=self.gt_value3,font=("Lucida Console", "10", "bold"), width=17, bg="#a9a9a9")
        label_mlsd_value1 = tk.Label(self.frame_vp, textvariable=self.mlsd_value1,font=("Lucida Console", "10", "bold"), width=17, bg="#a9a9a9")
        label_mlsd_value2 = tk.Label(self.frame_vp, textvariable=self.mlsd_value2,font=("Lucida Console", "10", "bold"), width=17, bg="#a9a9a9")
        label_mlsd_value3 = tk.Label(self.frame_vp, textvariable=self.mlsd_value3,font=("Lucida Console", "10", "bold"), width=17, bg="#a9a9a9")

        # vpのラベル配置
        label_GT.grid(row=0,column=1, columnspan=3, padx=5, pady=5)
        label_mlsd.grid(row=0,column=4, columnspan=3,padx=5, pady=5)

        label_vp1.grid(row=1,column=0,padx=2, pady=5)
        label_vp2.grid(row=2,column=0,padx=2, pady=5)
        label_vp3.grid(row=3,column=0,padx=2, pady=5)

        label_GT_value1.grid(row=1,column=1, columnspan=3,padx=1, pady=5)
        label_GT_value2.grid(row=2,column=1, columnspan=3,padx=1, pady=5)
        label_GT_value3.grid(row=3,column=1, columnspan=3,padx=1, pady=5)

        label_mlsd_value1.grid(row=1,column=4, columnspan=3,padx=1, pady=5)
        label_mlsd_value2.grid(row=2,column=4, columnspan=3,padx=1, pady=5)
        label_mlsd_value3.grid(row=3,column=4, columnspan=3,padx=1, pady=5)

    def file_dialog(self, event):
        selectDB=self.select_db.get()
        fTyp=[("Image File", "*.png;*.jpg;*.bmp")]
        if selectDB == 'YorkUrbanDB':
            iDir=os.path.join("./"+selectDB)
        elif selectDB == "Wireframe":
            iDir=os.path.join("./"+selectDB)
        file_name=tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        img_name=file_name
        if len(img_name)==0:
            self.img_name.set("Your selection has been canceled")
            self.img=None
            print("Your selection has been canceled")

        else:
            self.img = ImageTk.PhotoImage(self.keepAspectResize(Image.open(img_name),self.canvas_input_width, self.canvas_input_height))
            self.canvas_input.itemconfig(self.canvas_img_input, image=self.img)
            self.file_name.set(img_name)
            self.img_name.set("Image selected -> " + os.path.basename(img_name))
            print("Image selected -> " + os.path.basename(img_name))

    def save_image(self, event):
        selectDB=self.select_db.get()
        fTyp=[("Image File", "*.png;*.jpg;*.bmp")]
        if selectDB == 'YorkUrbanDB':
            iDir=os.path.join("./"+selectDB)
        elif selectDB == "Wireframe":
            iDir=os.path.join("./"+selectDB)
        file_name=tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
        print("Saved in",file_name)

    def keepAspectResize(self, img, width, height): # 画像のアスペクト比変更
                x_ratio = width / img.width
                y_ratio = height / img.height
                if x_ratio < y_ratio:
                    resize_size = (width, round(img.height * x_ratio))
                else:
                    resize_size = (round(img.width * y_ratio), height)
                resized_image = img.resize(resize_size)
                return resized_image

    def clear_result(self, event):
        #self.img=None
        self.mlsd_value1.set("0,0")
        self.mlsd_value2.set("0,0")
        self.mlsd_value3.set("0,0")
        self.img_output=None
        print("Result cleared")

    # スライダーの値を取得
    def slider_scroll_dist(self, str):
        dist_var=float(str)
        return dist_var

    def slider_scroll_score(self, str):
        score_var=float(str)
        return score_var

    def slider_scroll_len(self, str):
        len_var=float(str)
        return len_var

    def run(self, event):
        dist_val=self.slider_scroll_dist(self.scale_var_dist.get()) # distの値
        score_val=self.slider_scroll_score(self.scale_var_score.get()) # scoreの値
        len_val=self.slider_scroll_len(self.scale_var_len.get()) # line lengthの値
        if len(self.file_name.get())==0:
            print("wait")
        else:
            image_path=self.file_name.get() # 画像フォルダパス
            image_name=Path(image_path).stem #画像名
            image_width=Image.open(image_path).width # 画像の高さ
            image_height=Image.open(image_path).height # 画像の幅
            vp_mlsd, vp_gt, img_vp =test.test(score_val, dist_val, len_val, image_path, image_name, image_width, image_height)
            self.gt_value1.set(str(vp_gt[0,:])[1:-1].center(17))
            self.gt_value2.set(str(vp_gt[1,:])[1:-1].center(17))
            self.gt_value3.set(str(vp_gt[2,:])[1:-1].center(17))
            self.mlsd_value1.set(str(vp_mlsd[0,:])[1:-1].center(17))
            self.mlsd_value2.set(str(vp_mlsd[1,:])[1:-1].center(17))
            self.mlsd_value3.set(str(vp_mlsd[2,:])[1:-1].center(17))
            self.img_output = self.cv2_to_tk(img_vp)
            self.canvas_output.itemconfig(self.canvas_img_output, image=self.img_output)

    # resetボタンを押したとき
    def reset(self, event):
        self.scale_dist_thr.set(10)
        self.scale_score_thr.set(0.2)
        self.scale_len_thr.set(10)
        print("reset value")

    # All clearボタンを押したとき
    def all_clear(self, event):
        self.img_name.set("ALL cleared")
        self.file_name.set("")
        self.img=None
        self.img_output=None
        self.scale_dist_thr.set(10)
        self.scale_score_thr.set(0.2)
        self.scale_len_thr.set(10)
        self.gt_value1.set("0,0")
        self.gt_value2.set("0,0")
        self.gt_value3.set("0,0")
        self.mlsd_value1.set("0,0")
        self.mlsd_value2.set("0,0")
        self.mlsd_value3.set("0,0")
        print("all clear")

    # quitボタンを押したとき
    def quit(self):
        self.master.destroy()
        sys.exit()

    def cv2_to_tk(self, cv2_image):
        # BGR -> RGB
        rgb_cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # NumPy配列からPIL画像オブジェクトを生成
        pil_image = self.keepAspectResize(Image.fromarray(rgb_cv2_image),self.canvas_output_width, self.canvas_output_height)

        # PIL画像オブジェクトをTkinter画像オブジェクトに変換
        tk_image = ImageTk.PhotoImage(pil_image)

        return tk_image

def main():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()
