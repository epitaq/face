import cv2
import mediapipe as mp
import numpy as np

def eye_point (cap):
    """
    face_maskに追加
    戻り値の[0]がマスク用の白黒画像
    [1]がメインの画像
    """
    #  目のインデックス
    LEFT_IRIS = [474,475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    #  処理
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as face_mesh:
        rgb_frame = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        #  サイズの取得
        img_h, img_w, _ = cap.shape
        #  データの取得
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            z_value = np.array([p.z for p in results.multi_face_landmarks[0].landmark])
            #  目の座標 
            (l_x, l_y), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_x, r_y), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            l_center = np.array([l_x, l_y], dtype=np.int32)
            r_center = np.array([r_x, r_y], dtype=np.int32)
            #描画を黒背景にする  ##########
            #  目の描画 
            cap = np.zeros_like(cap)  
            cv2.circle(cap, center=l_center, radius=int(l_radius), color=(50,30,30), thickness=-1)
            cv2.circle(cap, center=r_center, radius=int(r_radius), color=(50,30,30), thickness=-1)
            #  目の中の座標
            l_iris = mesh_points[473]
            r_iris = mesh_points[468]
            cv2.circle(cap, center=l_iris, radius=int(l_radius/2), color=(10,10,10), thickness=-1)
            cv2.circle(cap, center=r_iris, radius=int(r_radius/2), color=(10,10,10), thickness=-1)
        #  二値化して黒白反転
        thresh = np.where(cap != 0, 0, 255).astype(np.uint8)
        return thresh, cap




def pic():
    img = cv2.imread('photo.jpg')
    img = eye_point(img)
    cv2.imshow("img_test", img[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def movie ():
    camera = cv2.VideoCapture(0)
    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    while True:
        ret, frame = camera.read()
        frame = eye_point(frame)[1]
        cv2.imshow('camera', frame)
        # キー操作があればwhileループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    # 撮影用オブジェクトとウィンドウの解放
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #pic()
    movie()