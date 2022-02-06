import cv2
import mediapipe as mp
import numpy as np


def hand_point (cap):
    #mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        #  座標
        img_h, img_w, _ = cap.shape
        #点をうつ
        #  背景を黒にする  #################
        cap = np.zeros_like(cap)
        if results.multi_hand_landmarks:
            #  手の数
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                z_list = [lm.z for lm in hand_landmarks.landmark]
                #print(z_list)
                z_min = min(z_list)
                z_max = max(z_list)
                #  pointの数
                for lm in hand_landmarks.landmark:
                    lm_xy = (int(lm.x * img_w), int(lm.y * img_h))
                    lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
                    lm_z = 255 - lm_z
                    #print(lm_z)
                    cv2.circle(cap, center=lm_xy, radius=6, color=(255,120,255), thickness=-1)
        #  二値化して黒白反転
        thresh = np.where(cap != 0, 0, 255).astype(np.uint8)
        return thresh, cap



def movie ():
    # カメラCh.(ここでは0)を指定
    camera = cv2.VideoCapture(0)
    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    while True:
        # フレームを取得
        ret, frame = camera.read()
        frame = hand_point(frame)
        # フレームを画面に表示
        cv2.imshow('camera', frame[1])
        # キー操作があればwhileループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    # 撮影用オブジェクトとウィンドウの解放
    camera.release()
    cv2.destroyAllWindows()

def pic():
    img = cv2.imread('photo.jpg')
    img = hand_point(img)
    cv2.imshow("img_test", img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #pic()
    movie()