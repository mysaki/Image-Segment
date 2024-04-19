import matplotlib.pyplot as plt
i=0
x=[]
y=[]
def create_img(i):
    plt.clf()  #清除上一幅图像
    fig=plt.imread('./img/train/observe_{}.jpg'.format(i))
    plt.imshow(fig)
    plt.pause(0.01)  # 暂停0.01秒
    # plt.ioff()  # 关闭画图的窗口
    
if __name__ == '__main__':
    for i in range(1000):
        create_img(i)
