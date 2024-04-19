import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
import logging
from flow import get_flow
sys.setrecursionlimit(100000) #例如这里设置为十万 

class Transfer():
    def __init__(self):
        self.init_img = None # 初始图片，用来打参考中心点
        self.target_center=[0,0]
        self.boder_box=[]
        self.click_count=0
        self.is_labeled =False
        self.candidates=[]
        self.res=30 #阈值，判断与上一个中心点的偏离是否超出允许范围
        self.step=0
        self.log()

    def log(self):
        # 第一步，创建一个logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO

        # 第二步，创建一个handler，用于写入日志文件
        logfile = './log.txt'
        fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关

        # 第三步，再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)   # 输出到console的log等级的开关

        # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 第五步，将logger添加到handler里面
        logger.addHandler(fh)
        logger.addHandler(ch)


    def draw_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # cv2.circle(self.init_img,(87,104),5,(255,0,0),-1)
            # cv2.circle(self.init_img,(157,186),5,(255,0,0),-1)
            if self.click_count == 0:
                cv2.circle(self.init_img,(x,y),5,(0,0,255),-1)
                self.target_center=[x,y]
                self.click_count+=1
                print("center point:",self.target_center)
            elif self.click_count == 1:
                cv2.circle(self.init_img,(x,y),5,(255,0,0),-1)
                self.boder_box.append([x,y])
                self.click_count+=1
                print("first point of boder box:",self.boder_box[0])
            elif self.click_count == 2:
                cv2.circle(self.init_img,(x,y),5,(255,0,0),-1)
                self.boder_box.append([x,y])
                print("second point of boder box:",self.boder_box[1])
                self.click_count+=1

    def numIslands(self, grid) -> int: #递归的求法太耗时了
        def zero (grid,x,y,max_x,max_y):
            grid[x][y]=False
            self.amount+=1
            if y < self.left:
                self.left = y
            if y > self.right:
                self.right = y
            if x < self.top:
                self.top = x
            if x > self.bottom:
                self.bottom = x
            if x+1 < max_x and grid[x+1][y]==True:
                grid= zero(grid,x+1,y,max_x,max_y)
            if y+1 < max_y and grid[x][y+1]==True:
                grid= zero(grid,x,y+1,max_x,max_y)
            if x-1 >=0 and grid[x-1][y]==True:
                grid= zero(grid,x-1,y,max_x,max_y)
            if y-1 >=0 and grid[x][y-1]==True:
                grid= zero(grid,x,y-1,max_x,max_y)

            if y-1 >=0 and x-1>=0 and grid[x-1][y-1]==True:
                grid= zero(grid,x-1,y-1,max_x,max_y)

            if y-1 >=0 and x+1< max_x and grid[x+1][y-1]==True:
                grid= zero(grid,x+1,y-1,max_x,max_y)

            if y+1 < max_y and x+1< max_x and grid[x+1][y+1]==True:
                grid= zero(grid,x+1,y+1,max_x,max_y)

            if y+1 < max_y and x-1>=0 and grid[x-1][y+1]==True:
                grid= zero(grid,x-1,y+1,max_x,max_y)

            return grid

        count = 0
        self.candidates=[]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==True:
                    count+=1
                    self.left=len(grid)
                    self.right=0
                    self.top=len(grid)
                    self.bottom=0
                    self.amount=0
                    grid=zero(grid,i,j,len(grid),len(grid[0]))
                    self.candidates.append([j,i,self.amount,self.left,self.top,self.right,self.bottom])
                    
        return count
    
    def get_init_target(self,img): # 打初始参考中心点
        self.init_img=img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_point)
        while True:
            cv2.imshow('image',self.init_img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow('image')

    def get_info(self,img):
        # if not self.is_labeled: #最开始需要进行标记
        #     # self.get_init_target(img)
        #     self.is_labeled=True
        # else:#如果已经标记过了，那就更新边界框和中心点
        #     self.get_new_info(img)
        self.is_labeled=True
        self.get_new_info(img)
        angle_info=np.arctan([(self.target_center[0]-img.shape[0]/2)/(img.shape[0]/2)*np.tan(np.radians(28.5))])
        dist_info=np.abs(self.boder_box[0][0]-self.boder_box[1][0])*np.abs(self.boder_box[0][1]-self.boder_box[1][1])
        return angle_info,dist_info
    
    def get_boder(self,img): # 获得像素块的边界
        self.init_img=img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_point)
        while True:
            
            cv2.imshow('image',self.init_img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow('image')
    
    def calcu_dist(self,n1,n2):
        dist=np.sqrt((n1[0]-n2[0])**2+(n1[1]-n2[1])**2)
        return dist

    def calcu_area(self,boder_box):
        h=-boder_box[0][1]+boder_box[1][1]
        w=-boder_box[0][0]+boder_box[1][0]
        return h*w

    def check_info(self,old_center,old_boder,new_center,new_boder,dist):
        final_center=[]
        final_boder=[[],[]]
        left_top_diff = self.calcu_dist(old_boder[0],new_boder[0])
        right_bottom_diff = self.calcu_dist(old_boder[1],new_boder[1])
        # logging.info("old_center:{},new_center:{}".format(old_center,new_center))
        # logging.info("old_boder:{},new_boder:{}".format(old_boder,new_boder))
        # print("old_center:",old_center,"new_center:",new_center)
        # print("old_boder:",old_boder,"new_boder:",new_boder)
        old_area=self.calcu_area(old_boder)
        new_area=self.calcu_area(new_boder)
        # print("old area:",old_area,"new area:",new_area,"change ratio:",new_area/old_area)
        # print("left_top_diff: ",left_top_diff,"right_bottom_diff: ",right_bottom_diff,"dist:",dist)
        # logging.info("area:{},area:{},reation:{}".format(old_area,new_area,new_area/old_area))
        # logging.info("left_top_diff:{},right_bottom_diff:{},dist:{}".format(left_top_diff,right_bottom_diff,dist))
        if dist >self.res:
            # print("center need to be fixed")
            # logging.info("center need to be fixed !")
            if left_top_diff >right_bottom_diff and right_bottom_diff <20:# 偏离较小的点也不能偏离太多
                # print("left_top changed too much !")
                # logging.info("left_top changed too much !")
                final_boder[1] = new_boder[1]
                h=-old_boder[0][1]+old_boder[1][1]
                w=-old_boder[0][0]+old_boder[1][0]
                final_boder[0] = [new_boder[1][0]-w,new_boder[1][1]-h]
            elif left_top_diff <= right_bottom_diff and left_top_diff <20:# 偏离较小的点也不能偏离太多
                # print("right_bottom changed too much !")
                # logging.info("right_bottom changed too much !")
                final_boder[0] = new_boder[0]
                h=-old_boder[0][1]+old_boder[1][1]
                w=-old_boder[0][0]+old_boder[1][0]
                final_boder[1] = [new_boder[0][0]+w,new_boder[0][1]+h]
            else:#如果两个点都偏离太多，就直接按原来的来
                # logging.info("keep unchanged !")
                final_boder=old_boder
            final_center=[final_boder[0][0]+(final_boder[1][0]-final_boder[0][0])/2,final_boder[0][1]+(final_boder[1][1]-final_boder[0][1])/2]
        else:
            # print("boder need to be fixed")
            # logging.info("boder need to be fixed !")
            #如果中心点改变不多的话，应考虑对整个框的大小进行修正
            final_center=new_center
            old_area=self.calcu_area(old_boder)
            new_area=self.calcu_area(new_boder)
            if new_area/old_area > 1.2 or  new_area/old_area <0.8 : #面积变化太大，表明估计有误
                h=-old_boder[0][1]+old_boder[1][1]
                w=-old_boder[0][0]+old_boder[1][0]
                final_boder=[[new_center[0]-w/2,new_center[1]-h/2],[new_center[0]+w/2,new_center[1]+h/2]]
            else:
                # logging.info("keep unchanged !")
                final_boder=new_boder
        # print("final_center:",final_center,"final_boder:",final_boder)
        return final_center,final_boder     
    def get_new_info(self,img):
        # logging.info("epoch - {} - step -{} ".format(epoch,self.step))
        """
        更新维护边界框以及中心点
        """
        process_img=img
        # print(process_img)
        # for i in range(len(process_img)):
        #     for j in range(len(process_img[0])):
        #         if process_img[i][j] == 'True' :
        #             process_img[i][j]=1.0
        #         else:
        #             process_img[i][j]=0.0
        self.numIslands(process_img.copy()) # 找到分割图像里每个像素块左上角的坐标,及每个像素块中的像素数量
        self.candidates.sort(key=lambda x: x[2],reverse=True)# 按每个像素块中的像素数量进行排序，表示该像素块是追踪目标的可能从大到小排列
        # print("candidates:",self.candidates) #
        temp_centers=[]
        new_flag=False

        if self.candidates != []:
            if self.is_labeled == False:
                new_boder_box=[[self.candidates[0][3],self.candidates[0][4]],[self.candidates[0][5],self.candidates[0][6]]]
                self.target_center=[new_boder_box[0][0]+(new_boder_box[1][0]-new_boder_box[0][0])/2,new_boder_box[0][1]+(new_boder_box[1][1]-new_boder_box[0][1])/2]
                self.boder_box=new_boder_box
                self.is_labeled = True
            else:
                for candidate in self.candidates:
                    # print("candidate:",candidate)
                    new_boder_box=[[candidate[3],candidate[4]],[candidate[5],candidate[6]]]
                    # 用border_box的中心点作为新的中心点
                    new_center=[new_boder_box[0][0]+(new_boder_box[1][0]-new_boder_box[0][0])/2,new_boder_box[0][1]+(new_boder_box[1][1]-new_boder_box[0][1])/2]
                    # while True:
                    #     show_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                    #     # cv2.circle(show_img,(215,96),5,(0,255,255),-1)
                    #     cv2.circle(show_img,(int(new_center[0]),int(new_center[1])),5,(255,255,0),-1)
                    #     cv2.circle(show_img,(int(candidate[3]),int(candidate[4])),5,(255,0,255),-1)
                    #     cv2.circle(show_img,(int(candidate[5]),int(candidate[6])),5,(255,0,255),-1)
                    #     cv2.imshow('result',show_img)
                    #     if cv2.waitKey(20) & 0xFF == 27:
                    #         break

                    dist=np.sqrt((new_center[0]-self.target_center[0])**2+(new_center[1]-self.target_center[1])**2)
                    if self.step == 0:
                        self.target_center=new_center
                        self.boder_box=new_boder_box
                        return
                    # print("dist:",dist)
                    if int(dist) < self.res: #如果满足要求，则作为新的中心点
                        # print("Get new center !")
                        old_center=self.target_center
                        old_boder=self.boder_box

                        self.target_center,self.boder_box=self.check_info(old_center,old_boder,new_center,new_boder_box,dist)
                        new_flag=True
                        break
                    else:
                    # print("Get center candidate!")
                        temp_centers.append([new_center,new_boder_box,dist]) # 不满足要求则加入到备选集合中
                if not new_flag:
                    temp_centers.sort(key=lambda x: x[2])
                    # print("temp_centers:",temp_centers)
                    old_center=self.target_center
                    old_boder=self.boder_box
                    self.target_center,self.boder_box=self.check_info(self.target_center,self.boder_box,new_center,new_boder_box,dist)
                # print("new_center:",self.target_center)
                # print("boder_box:",self.boder_box)
                # show_img=np.transpose(original_img,(1,2,0))
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale=0.5
                # thickness=2
                # text= str((self.target_center[0]-img.shape[0]/2)/(img.shape[0]/2))
                # color=(0,0,255)

                # show_img=cv2.cvtColor(show_img,cv2.COLOR_GRAY2BGR)
                
                # cv2.imwrite('./detect_result/{}/original_{}.jpg'.format(epoch,self.step),show_img*255)
                
                # if os.path.exists('./detect_result/{}'.format(epoch)) == False:
                #     os.mkdir('./detect_result/{}'.format(epoch))
 
                # cv2.putText(show_img,text, (int(self.target_center[0])+10, int(self.target_center[1])+10), font, font_scale, color, thickness)
                # cv2.circle(show_img,(int(self.target_center[0]),int(self.target_center[1])),5,(0,0,255),-1)
                # cv2.circle(show_img,(int(self.boder_box[0][0]),int(self.boder_box[0][1])),5,(255,0,255),-1)
                # cv2.circle(show_img,(int(self.boder_box[1][0]),int(self.boder_box[1][1])),5,(255,0,255),-1)
                # cv2.circle(show_img,(int(old_center[0]),int(old_center[1])),5,(0,255,255),-1)
                # cv2.circle(show_img,(int(old_boder[0][0]),int(old_boder[0][1])),5,(255,255,0),-1)
                # cv2.circle(show_img,(int(old_boder[1][0]),int(old_boder[1][1])),5,(255,255,0),-1)
                # cv2.imwrite('./detect_result/{}/{}.jpg'.format(epoch,self.step),show_img*255)
                self.step+=1
        else:
            # print("Target not found!")
            self.target_center=[128,128]
            self.boder_box =[[90,90],[200,200]]

if __name__ == '__main__':
    trans=Transfer()
    for m in range(1,1315):
        print(m)
        
        gray_img=cv2.imread('./sim_imgs/test/result_{}.jpg'.format(m-1),cv2.IMREAD_GRAYSCALE)
        gray_img=cv2.resize(gray_img,(256,256))
        temp=gray_img.copy()
        for i in range(len(gray_img)):
            for j in range(len(gray_img[0])):
                if gray_img[i][j] ==0:
                    temp[i][j]=False
                else:
                    temp[i][j]=True
        angle_info,dist_info=trans.get_info(temp)
        print(trans.target_center)
        gray_img=cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)


        img=cv2.imread('./sim_imgs/train/observe_{}.jpg'.format(m))
        pre_img=cv2.imread('./sim_imgs/train/observe_{}.jpg'.format(m-1))
        predict_center=get_flow(pre_img,img,trans.target_center)
        print("Flow Predict:",predict_center)
        now_img=cv2.imread('./sim_imgs/test/result_{}.jpg'.format(m),cv2.IMREAD_GRAYSCALE)
        now_img=cv2.resize(now_img,(256,256))
        temp=now_img.copy()
        for i in range(len(now_img)):
            for j in range(len(gray_img[0])):
                if now_img[i][j] ==0:
                    temp[i][j]=False
                else:
                    temp[i][j]=True
        angle_info,dist_info=trans.get_info(temp)
        now_img=cv2.cvtColor(now_img,cv2.COLOR_GRAY2BGR)
        cv2.circle(now_img, (int(predict_center[0] ), int(predict_center[1] )), 5, (0, 0, 255), -1)
        cv2.circle(now_img, (int(trans.target_center[0] ), int(trans.target_center[1] )), 5, (255, 0, 255), -1)
        plt.imshow(now_img,cmap='gray')
        plt.show()

        # print("img:seg_{}.png".format(m))
        
        # # print("new_center:",trans.target_center)
        # print("angle_info:",np.degrees(angle_info),"dist_info",dist_info)
        # cv2.namedWindow('result')
        # while True:
        #     show_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #     cv2.circle(show_img,(int(trans.target_center[0]),int(trans.target_center[1])),5,(0,0,255),-1)
        #     cv2.circle(show_img,(int(trans.boder_box[0][0]),int(trans.boder_box[0][1])),5,(255,0,255),-1)
        #     cv2.circle(show_img,(int(trans.boder_box[1][0]),int(trans.boder_box[1][1])),5,(255,0,255),-1)
        #     cv2.imshow('result',show_img)
        #     if cv2.waitKey(20) & 0xFF == 27:
        #         break
        # cv2.destroyWindow('result')

        

        




