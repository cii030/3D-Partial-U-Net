import numpy as np
from skimage.morphology import square,cube,dilation,erosion
from mayavi import mlab

coefficient = 1 #max系数

# data = np.load(r'D:/Display/dice_record/Gau_noGau/qualitative/3D/patient682_92484652_11.npy')
# data = np.load('test-ct.npy')
# data1 = data[2]
# data2 = data[3]
lst = ['UNet3d', 'SegResNet', 'AttUNet', 'UNETR', 'MedNext','UXNet', 'PUNet-innorm-lbr' ]
data1 = np.load('mask_2610.npy')
data2 = np.load(f'./{lst[6]}/2610.npy')
data1 = data1[0]
data2 = data2[0]
# np.save('data2_ori', data2)  # 保存data1为data1.npy文件

# for i in range(data1.shape[2]):
#     data13=erosion(data1[:,:,i], square(3))
#     data1[:,:,i]=data1[:,:,i]-data13
#
# for i in range(data2.shape[2]):
#     data3=erosion(data2[:,:,i], square(3))
#     data2[:,:,i]=data2[:,:,i]-data3

data11=erosion(data1, cube(3))         # 求轮廓，求面与面之间的distance
data1=data1-data11
data22=erosion(data2, cube(3))         # 求轮廓，求面与面之间的distance
data2=data2-data22

# np.save('data2_rev', data2)  # 保存data1为data1.npy文件
x, y, z = data1.shape    # 获取矩阵形状
temp1 = []
temp2 = []
# 通过循环得出data1和data2中值为1的坐标点并用temp1和temp2储存起来
for i in range(x):
    for j in range(y):
        for k in range(z):
            if data1[i, j, k] == 1:
                temp1.append([i, j, k])
            if data2[i, j, k] == 1:
                temp2.append([i, j, k])
            if data1[i, j, k] == 0:    # 将data1中值为0的点赋值为负值
                data1[i, j, k] = -1

print('temp1', np.sum(temp1))

for x in temp1:
    temp = []
    for y in temp2:
        num = np.sqrt((x[0]-y[0]) ** 2 + (x[1]-y[1]) ** 2 + (x[2]-y[2]) ** 2)   # 计算距离并用temp储存起来
        temp.append(num)
    data1[x[0], x[1], x[2]] = min(temp) # 找出最小的值
max_num = np.unique(data1)[-1]     # sort and find the last one
print('max_num', max_num)
# data1 = data1 / max_num * coefficient #colorbar更改范围

data1[25, 31, 24]=4         #  919747421                   #  temp1
data1 = data1

# np.save('data2', data1)  # 保存data1为data1.npy文件

a = []
b = []
c = []
s = []
x, y, z = data1.shape
# 获得点向量和对应值
for i in range(x):
    for j in range(y):
        for k in range(z):
            if data1[i, j, k] >= 0:
                a.append(i)
                b.append(j)
                c.append(k)
                s.append(data1[i, j, k])
a = np.array(a)/50
b = np.array(b)/50
c = np.array(c)/50
s = np.array(s)
points = mlab.points3d(a, b, c, s, scale_mode='none', scale_factor=0.3) #画图
# mlab.colorbar(title='Distance', orientation='vertical', nb_labels=3) #画diastance颜色表
cb=mlab.colorbar(orientation='vertical', nb_labels=3) #画diastance颜色表
cb.scalar_bar.unconstrained_font_size = True
cb.label_text_property.font_family = 'times'
cb.label_text_property.bold = 1
cb.label_text_property.font_size=30
mlab.show()