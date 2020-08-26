import tensorflow as tf 


class Conv_block():
    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(Conv_block, self).__init__()
        self.conv = tf.keras.Sequential((
            tf.keras.layers.Conv2D(filters = ch_in, kernel_size = 3, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Conv2D(filters = ch_out, kernel_size = 3, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU()
        ))

class Up_conv():
    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(Up_conv, self).__init__()
        self.up = tf.keras.Sequential((
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters = ch_out, kernel_size = 3, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU()
        ))
class U_Net():
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        self.Maxpool = tf.keras.layers.MaxPool2D()

        self.Conv1 = Conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = Conv_block(ch_in=64,ch_out=128)
        self.Conv3 = Conv_block(ch_in=128,ch_out=256)
        self.Conv4 = Conv_block(ch_in=256,ch_out=512)
        self.Conv5 = Conv_block(ch_in=512,ch_out=1024)

        self.Up5 = Up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = Conv_block(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = Conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = Up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = Conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = Up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = Conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = tf.keras.layers.Conv2D(1, kernel_size=output_ch, use_bias=False)
    def build(self, x):
        # encoding path
        x1 = self.Conv1.conv(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2.conv(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3.conv(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4.conv(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5.conv(x5)

        # decoding + concat path
        d5 = self.Up5.up(x5)
        d5 = tf.concat((x4,d5), axis=-1)
        d5 = self.Up_conv5.conv(d5)
        
        d4 = self.Up4.up(d5)
        d4 = tf.concat((x3,d4),axis=-1)
        d4 = self.Up_conv4.conv(d4)

        d3 = self.Up3.up(d4)
        d3 = tf.concat((x2,d3),axis=-1)
        d3 = self.Up_conv3.conv(d3)

        d2 = self.Up2.up(d3)
        d2 = tf.concat((x1,d2), axis=-1)
        d2 = self.Up_conv2.conv(d2)

        d1 = self.Conv_1x1(d2)

        return d1

