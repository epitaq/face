import cv2
from matplotlib import scale
#import mediapipe as mp
import numpy as np
import time

#from eye import eye_point
from hand import hand_point
from face_mask import face_point



def movie ():
    time0 = time.time()
    # カメラCh.(ここでは0)を指定
    camera = cv2.VideoCapture(0)
    #  元となる画像
    ret, img = camera.read()
    originate_img = np.zeros_like(img)
    upload_img = cv2.imread('random1.png')
    originate_img = cv2.resize(upload_img, dsize=(img.shape[1], img.shape[0]))
    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    while True:
        #  フレームレートの取得
        time1 = time.time()
        fps = round(1 / (time1 - time0), 2)
        time0 = time.time()
        # フレームを取得
        ret, img = camera.read()
        #main_img = img
        main_img = originate_img
        #  合成
        #  eye faceに合成
        #  face
        face_result = face_point(img)
        face_and = cv2.bitwise_and(main_img, face_result[0])
        main_img = cv2.add(face_and, face_result[1])
        #  hand
        hand_result = hand_point(img)
        hand_and = cv2.bitwise_and(main_img, hand_result[0])
        main_img = cv2.add(hand_and, hand_result[1])
        #  fpsの表示
        fps_img = cv2.putText(main_img, text=str(fps)+'fps', org=(0,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10,255,10))
        #img = cv2.bitwise_or(img, fps_img)
        img = fps_img
        # フレームを画面に表示
        cv2.imshow('camera', img)
        # キー操作があればwhileループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    # 撮影用オブジェクトとウィンドウの解放
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #pic()
    movie()