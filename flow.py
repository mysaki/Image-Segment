import cv2
import numpy as np
import matplotlib.pyplot as plt
def get_flow(frame1,frame2,center):
    # 读取图像
    # frame1 = cv2.imread('./sim_imgs/train/observe_0.jpg')
    # frame2 = cv2.imread('./sim_imgs/train/observe_1.jpg')
    
    # 转换为灰度图像
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev=prev_gray, next=next_gray, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # 创建一个用于可视化的空图像
    h, w = flow.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # 可视化光流
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            vx, vy = flow[y, x]
            cv2.line(vis, (x, y), (int(x + vx), int(y + vy)), (0, 255, 0), 1)
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
    # 显示结果
    plt.imshow(vis)
    plt.show()
    vx, vy = flow[int(center[1]),int(center[0])]
    return [ vx,vy]
    # cv2.circle(frame2, (int(center[0]),int(center[1])), 1, (0, 0, 255), -1)
    # cv2.circle(frame2, (int(center[0] + vx), int(center[1] + vy)), 1, (0, 255, 0), -1)


                  


if __name__ == '__main__':
    frame1 = cv2.imread('./sim_imgs/train/observe_24.jpg')
    frame2 = cv2.imread('./sim_imgs/train/observe_25.jpg')
    print(get_flow(frame1, frame2,[126,108]))
