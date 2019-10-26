# 如何拍照：
* 拍照的时候需要站在地上（不是草上不是桌子上）。
* 障碍物必须是在地上的（不是桌子上的，他们不在桌上走路）。
* 由于我们的model是一个Fully Convolutional Neural Network，照片可以是任何大小，但是用app拍的时候必须横过来。
* 可以考虑给一个obstacle从不同的角度、不同的光线环境下、以及不同的远近进行拍摄。这样既可以提前完成指标，也可以确保model够可靠。
* 目前，我们的model会把阴影识别成obstacle，所以如果碰到这种情况请拍下来。

# 如何label：
* 障碍物的框需要紧紧的包裹住这个障碍物 (The box should tightly bounding the obstacle) 。也就是说，box的四条边每一条都与object接触（非常非常重要，大型的dataset都是以pixel为精度的，我们只需要tight就行了）
* 需要标出7米以内的所有的障碍物。如果在室外，远处明显有障碍物的也可进行标记。
* 定义一下不同的label
  * Obstacle - 顾名思义
    * 需要照各式各样的Obstacle，大的小的。例如桌子，不同的桌子，不同颜色的桌子，半张桌子一整张桌子。
    * 墙也算obstacle啊。
  * Edge - 人行道的边缘，就是从人行道走到机动车道的那个边缘"台阶"
    * 一条edge用一个box框。如果edge是直的，这条edge应该是这个box的对角线。
    * 不要一条edge用10个box。用一个，即使这一个大得撑满了照片。
  * Pothole - 坑，通常是被砸出来的，或者井盖没盖好之类的
    * La Jolla的路太好了，这种情况超级少
  * Uplift - 人行道是一块一块concrete的，刚铺的时候concrete与concrete之间的缝隙是在一个平面上的，但是随着岁月的累积，一块会比另一块高，这个高出的一段叫做uplift，其看似平坦的路却可能被绊倒。

# 强调一下label很重要：
目前的三百多张照片，在修正label之前，train出来的model什么都识别不了，loss只converge了一点点。Label的精准度很重要。

大家看一下已经修正好label的图片，get an idea of what I am talking about。

我们现在有三百多张照片。训练出来的model还不错。但是data里没有的obstacle就识别不出来。To obtain better accuracy, we need data. 

