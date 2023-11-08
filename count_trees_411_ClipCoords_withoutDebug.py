import os
import torch
import csv
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import random

# 座標をスケールする関数
def scale_coords(img1_shape, coords, img0_shape):
    # 画像の拡大率を計算
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    # パディング量を計算
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2
    # 座標を修正
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, [0, 2]] /= gain
    coords[:, [1, 3]] /= gain

    # 座標を画像のサイズにクリップ
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])

    # 座標を整数に丸める
    coords = coords.round().astype(int)

    return coords

# バウンディングボックスを描画する関数
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    im0 = img.copy()
    tl = line_thickness or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # 線の太さ
    color = color or [random.randint(0, 255) for _ in range(3)]  # 色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 座標
    # バウンディングボックスを描画
    cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # ラベルが存在する場合、ラベルを描画
    if label:
        tf = max(tl - 1, 1)  # フォントの太さ
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # 塗りつぶし
        cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im0

# 木を検出する関数（木の本数を返すように変更）
def detect_tree(input_path, output_path):
    # YOLOv5モデルをロード
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\natuz\\yolov5\\runs\\train\\exp9\\weights\\best.pt')

    # 画像をロード
    img0 = Image.open(input_path)
    img = np.array(img0)

    # 物体検出
    results = model(img)
    det = results.xyxy[0].cpu().numpy()  # 検出された座標を取得

    # 元の画像の寸法を取得（height, width）
    img0_shape = img0.size[::-1]  # PILのsizeは(width, height)なので逆にする

    # 検出された座標を元の画像サイズにスケール
    det = scale_coords(img.shape[1:], det, img0_shape).round()  # ここでimg0.size[::-1]に変更

    # 検出された木の本数
    tree_count = det.shape[0]

    for *xyxy, conf, cls in det:
        # (x_center, y_center, width, height)を(x1, y1, x2, y2)に変換
        x_center, y_center, width, height = xyxy
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        label = f"{results.names[int(cls)]} {conf:.2f}"
        # 描画された画像をimgに更新する
        img = plot_one_box(xyxy, img, label=label, color=[255, 0, 0], line_thickness=3)

    # BGR形式に変換して保存
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return tree_count  # 木の本数を返す

# 全ての画像を処理し、木の本数をCSVに出力する関数
def process_all_images_and_output_csv(input_dir, output_dir, csv_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSVファイルを開く
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ファイル名', '木の本数', '補足'])  # ヘッダーを書き込む

        # 入力ディレクトリ内の全ての画像を処理
        for img_path in input_dir.glob("*.jpg"):
            output_path = output_dir / img_path.name
            tree_count = detect_tree(str(img_path), str(output_path))  # 木の本数を取得

            # CSVに書き込む
            if img_path.stem.isdigit() and 1 <= int(img_path.stem) <= 50:
                writer.writerow([img_path.name, tree_count, 'アノテーションに利用'])
            else:
                writer.writerow([img_path.name, tree_count, ''])

input_dir = r"C:\Users\natuz\OneDrive\デスクトップ\Programming\JPT_Intern\kenjimasuda_jpt_firsttask\doc\existing_data\images"
output_dir = r"C:\Users\natuz\OneDrive\デスクトップ\Programming\JPT_Intern\kenjimasuda_jpt_firsttask\doc\existing_data\output_images"
csv_path = r"C:\Users\natuz\OneDrive\デスクトップ\Programming\JPT_Intern\kenjimasuda_jpt_firsttask\doc\results\results.csv"
process_all_images_and_output_csv(input_dir, output_dir, csv_path)
