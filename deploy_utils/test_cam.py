# 保存为 test_camera_final.py
import cv2
import time

# 初始化摄像头，强制MJPG+低参数
cap = cv2.VideoCapture("/dev/video6")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 预热+读取10帧，验证稳定性
success_count = 0
for i in range(10):
    ret, frame = cap.read()
    time.sleep(0.1)  # 等待帧就绪
    if ret and frame is not None:
        success_count += 1
        print(f"第{i+1}帧：成功，尺寸{frame.shape}")
    else:
        print(f"第{i+1}帧：失败")

print(f"\n总成功数：{success_count}/10")
if success_count > 5:
    print("摄像头硬件正常，问题在lerobot的读取逻辑！")
else:
    print("摄像头硬件/驱动仍有问题，建议换摄像头/USB口！")

cap.release()
cv2.destroyAllWindows()
