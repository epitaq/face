from termios import VWERASE
import cv2
import mediapipe as mp
import numpy as np

from face_mesh_lst import face_mesh_lst

def face_point (cap):
    mp_face_mesh = mp.solutions.face_mesh
    #  ポリゴンのリスト
    #face_mesh_lst = list(mp.python.solutions.face_mesh_connections.FACEMESH_TESSELATION)
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as face_mesh:
        rgb_frame = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        #  サイズの取得
        img_h, img_w, _ = cap.shape
        #  データの取得
        if results.multi_face_landmarks:
            #  X,Yの座標をリスト化
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            #  zをリスト化
            z_value = np.array([p.z for p in results.multi_face_landmarks[0].landmark])
            #  描画を黒背景にする  ##############
            cap = np.zeros_like(cap)
            thresh = np.zeros_like(cap)
            for i in range(0, len(face_mesh_lst), 3):
                vertex = [face_mesh_lst[i][0], face_mesh_lst[i][1], face_mesh_lst[i+1][1]]
                #0になるとマスク処理に失敗するから防ぐ
                color_r = vertex[0]%254 + 1
                color_g = vertex[1]%254 + 1
                color_b = vertex[2]%254 + 1
                cv2.fillConvexPoly(cap, mesh_points[vertex], color=(color_r,color_g,color_b), lineType=cv2.LINE_8, shift=0) 
        #  二値化して黒白反転
        thresh = np.where(cap != 0, 0, 255).astype(np.uint8)
        return thresh, cap


def pic():
    img = cv2.imread('photo.jpg')
    img = face_point(img)
    cv2.imshow("img_test", img[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def movie ():
    camera = cv2.VideoCapture(0)                # カメラCh.(ここでは0)を指定
    # 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
    while True:
        ret, frame = camera.read()              # フレームを取得
        frame = face_point(frame)
        cv2.imshow('camera', frame[0])             # フレームを画面に表示
        # キー操作があればwhileループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    # 撮影用オブジェクトとウィンドウの解放
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #pic()
    movie()