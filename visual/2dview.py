import numpy as np
import matplotlib.pyplot as plt


def view(ct, data_mask, data_pred, slice, name):
    ct = ct[:, :, slice]
    data_mask = data_mask[:, :, slice]
    data_pred = data_pred[:, :, slice]
    img_max = ct.max()
    img_min = ct.min()
    out_img = (ct - img_min) / (img_max - img_min)

    out_mask = data_mask

    out_pred = data_pred

    image_mask = out_mask * 3

    val_outputs_pre = out_pred
    overlap = image_mask - val_outputs_pre
    overlap_idx = np.where(overlap == 2)

    # --------------自定义颜色--------------------------------------
    #三个参数分别为红色、绿色、蓝色的深浅，取值0~255，0为透明
    custom_color = [255, 255, 255]
    # ------------------------------------------------------------

    # R channel
    img_where = np.where(image_mask == 3)
    img = np.zeros((out_pred.shape[0], out_pred.shape[1]))  # shape
    img[img_where] = 1

    # G channel
    val_where = np.where(val_outputs_pre == 1)
    val = np.zeros((out_pred.shape[0], out_pred.shape[1]))  # shape
    val[val_where] = 1

    # B channel
    mid = np.zeros((out_pred.shape[0], out_pred.shape[1]))  # shape
    mid[overlap_idx] = 1

    # R-B
    R_B = img - mid
    # G-B
    G_B = val - mid

    # ----------------------------------创建显示数组----------------------------------
    show_array = np.zeros((out_pred.shape[0], out_pred.shape[1], 3))  # shape

    # 给mid赋值，蓝色[0, 0, 255]
    show_array[:, :, 2][overlap_idx] = custom_color[2]
    # 此处↑ show_array[:, :, 2]改为[:, :, 1]->mid区域为绿色，改为[:, :, 0]->mid区域为红色

    # 给R-B赋值，绿色[0, 255, 0]
    overlap_rb = np.where(R_B == 1)  # GT区域
    show_array[:, :, 1][overlap_rb] = custom_color[1]
    # 此处↑ show_array[:, :, 1]改为[:, :, 0]->GT区域为红色，改为[:, :, 2]->GT区域为蓝色

    # 给G-B赋值，红色[255, 0, 0]
    overlap_gb = np.where(G_B == 1)  # segmentation result区域
    show_array[:, :, 0][overlap_gb] = custom_color[0]
    # 此处↑ show_array[:, :, 0]改为[:, :, 1]->分割区域为绿色，改为[:, :, 2]->分割区域为蓝色
    # show_array第三维中0，1，2必须全部出现
    # ------------------------------------------------------------------------------

    # 读取原图
    image = out_img
    image_3 = np.expand_dims(image, axis=2).repeat(3, axis=2)

    # 利用opencv中的add函数叠加图像
    image_add = image_3 + show_array

    # 设定一个比率，修改叠加图像的透明度
    ratio = 0.002  # ratio = 0.0008
    image_add = image_3 * (1 - ratio) + show_array * ratio

    # 存储图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'wspace': 0})
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax1.imshow(out_img, cmap='gray')
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax2.imshow(image_add, interpolation='none')
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax3.imshow(show_array)
    plt.tight_layout(pad=0)

    # 展示图像
    plt.show()
    fig.savefig(name)


data = np.load('img_9153.npy')
data1 = np.load('mask_9153.npy')
data2 = np.load('./PUNet-innorm-lbr/9153.npy')
data = data[0]
data1 = data1[0]
data2 = data2[0]
view(data, data1, data2, 24, 'punet9153')
# for i in range(48):
#     view(data, data1, data2, i, 'unet9153')
