import cv2
#import mediapipe as mp
import numpy as np
import time

#from eye import eye_point
from hand import hand_point
from face_mask import face_point


def movie ():
    # カメラCh.(ここでは0)を指定
    camera = cv2.VideoCapture(0)
    #  元となる画像

    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    time0 = time.time()
    while True:
        #  フレームレートの取得
        time1 = time.time()
        fps = round(1 / (time1 - time0), 2)
        time0 = time.time()
        # フレームを取得
        ret, img = camera.read()
        multi_img = np.zeros_like(img)
        #  合成
        # #  eye
        # eye_result = eye_point(img)
        # eye_and = cv2.bitwise_and(multi_img, eye_result[0])
        # multi_img = cv2.add(eye_and, eye_result[1])
        #  face
        face_result = face_point(img)
        face_and = cv2.bitwise_and(multi_img, face_result[0])
        multi_img = cv2.add(face_and, face_result[1])
        #  hand
        hand_result = hand_point(img)
        hand_and = cv2.bitwise_and(multi_img, hand_result[0])
        multi_img = cv2.add(hand_and, hand_result[1])
        #  fpsの表示
        fps_img = cv2.putText(multi_img, text=str(fps)+'fps', org=(0,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10,255,10))
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