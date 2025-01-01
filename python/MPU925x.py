#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import time
import math
import smbus
import RPi.GPIO as GPIO

Gyro  = [0,0,0]
Accel = [0,0,0]
Mag   = [0,0,0]
gyroOffset=[0,0,0]
magOffset=[0,0,0]
pitch = 0.0
roll  = 0.0
yaw   = 0.0
Ki = 1.0
Kp = 4.50
q0 = 1.0
q1=q2=q3=0.0
ADDR         = (0x68)

# /** Registers */
PWR_MGMT_1     = 0x6B
PWR_MGMT_2     = 0x6C
SMPLRT_DIV     = 0x19  
CONFIG         = 0x1A  
GYRO_CONFIG    = 0x1B  
ACCEL_CONFIG   = 0x1C
ACCEL_CONFIG_2 = 0x1D
INT_PIN_CFG    = 0x37
INT_ENABLE     = 0x38  

ACCEL_X = 0x3B  
ACCEL_Y = 0x3D  
ACCEL_Z = 0x3F  
GYRO_X  = 0x43  
GYRO_Y  = 0x45  
GYRO_Z  = 0x47  
TEMP    = 0x41  

MAG_X = 0x03
MAG_Y = 0x05
MAG_Z = 0x07
ST_1  = 0x02
ST_2  = 0x09
MAG_ADDRESS = 0x0C

MPU9255_REG_ID = 0x75 # identity of the device, 8 bit
MPU9250_ID     = 0x71 # identity of mpu9250 is 0x71
MPU9255_ID     = 0x73 # identity of mpu9255 is 0x73

# int pin
# INI_PIN = 23

class MPU925x:
    def __init__(self, address=ADDR):
        self.bus = smbus.SMBus(1)
        self.address = address

        self.ID = self.Read_Byte(MPU9255_REG_ID)
        if(self.ID != MPU9250_ID): # ID = 0x73
            print("ID = 0x%x"%self.ID)
            sys.exit()
        
        self.Write_Byte(PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        self.Write_Byte(PWR_MGMT_1, 0x01)
        time.sleep(0.1)
        self.Write_Byte(CONFIG, 0x03)
        self.Write_Byte(SMPLRT_DIV, 0x04)
        self.Write_Byte(GYRO_CONFIG, 0x18)    # 250dps, 
        self.Write_Byte(ACCEL_CONFIG, 0x01)   # 2g-scale
        self.Write_Byte(ACCEL_CONFIG_2, 0x03) # low pass
        self.Write_Byte(INT_PIN_CFG, 0x02)    # enable bus master bypass
        time.sleep(0.1)
        
        self.bus.write_byte_data(MAG_ADDRESS, 0x0A, 0x00)
        time.sleep(0.05)
        self.bus.write_byte_data(MAG_ADDRESS, 0x0A, 0x0F)
        time.sleep(0.05)
        self.bus.write_byte_data(MAG_ADDRESS, 0x0A, 0x00)
        time.sleep(0.05)
        self.bus.write_byte_data(MAG_ADDRESS, 0x0A, 0x12)
        time.sleep(0.1)
        
        self.readGyroOffset()
        self.magCalib()

    def Read_Byte(self, Addr):
        return self.bus.read_byte_data(self.address, Addr)

    def Write_Byte(self, Addr, val):
        self.bus.write_byte_data(self.address, Addr, val)
    
    def dataConv(self, data1, data2):
        value = data1 | (data2 << 8)
        if(value & (1 << 16 - 1)):
            value -= (1<<16)
        return value
    
    def accel(self):
        xh = self.Read_Byte(ACCEL_X)
        xl = self.Read_Byte(ACCEL_X + 1)
        x = self.dataConv(xl,xh)
        yh = self.Read_Byte(ACCEL_Y)
        yl = self.Read_Byte(ACCEL_Y + 1)
        y = self.dataConv(yl,yh)
        zh = self.Read_Byte(ACCEL_Z)
        zl = self.Read_Byte(ACCEL_Z + 1)
        z = self.dataConv(zl,zh)
        Buf = [x, y, z]
        return Buf
    
    def gyro(self):
        xh = self.Read_Byte(GYRO_X)
        xl = self.Read_Byte(GYRO_X + 1)
        x = self.dataConv(xl,xh) - gyroOffset[0]
        yh = self.Read_Byte(GYRO_Y)
        yl = self.Read_Byte(GYRO_Y + 1)
        y = self.dataConv(yl,yh) - gyroOffset[1]
        zh = self.Read_Byte(GYRO_Z)
        zl = self.Read_Byte(GYRO_Z + 1)
        z = self.dataConv(zl,zh) - gyroOffset[2]
        Buf = [x, y, z]
        return Buf
        
    def mag(self):
        # check data ready  
        self.bus.read_byte_data(MAG_ADDRESS, ST_1)
        xl = self.bus.read_byte_data(MAG_ADDRESS, MAG_X)
        xh = self.bus.read_byte_data(MAG_ADDRESS, MAG_X + 1)
        x = self.dataConv(xl,xh) - magOffset[0]
        yl = self.bus.read_byte_data(MAG_ADDRESS, MAG_Y)
        yh = self.bus.read_byte_data(MAG_ADDRESS, MAG_Y + 1)
        y = self.dataConv(yl,yh) - magOffset[1]
        zl = self.bus.read_byte_data(MAG_ADDRESS, MAG_Z)
        zh = self.bus.read_byte_data(MAG_ADDRESS, MAG_Z + 1)
        z = self.dataConv(zl,zh) - magOffset[2]
        self.bus.read_byte_data(MAG_ADDRESS, ST_2)
        Buf = [x, y, z]
        return Buf
    
    def readGyroOffset(self):
        s32TempGx = 0
        s32TempGy = 0
        s32TempGz = 0
        for i in range(0,32):
            self.gyro()
            s32TempGx += Gyro[0]
            s32TempGy += Gyro[1]
            s32TempGz += Gyro[2]
            time.sleep(0.01)
        gyroOffset[0] = s32TempGx >> 5
        gyroOffset[1] = s32TempGy >> 5
        gyroOffset[2] = s32TempGz >> 5
        
    def magCalib(self):
        '''
        magTemp=[0,0,0,0,0,0,0,0,0]
        print("\nkeep 10dof-imu device horizontal and it will read x y z axis offset value after 4 seconds\n")
        time.sleep(4)
        print("start read all axises offset value \n")
        self.mag()
        magTemp[0] = Mag[0]
        magTemp[1] = Mag[1]
        magTemp[2] = Mag[2]
        
        print("rotate z axis 180 degrees and it will read all axises offset value after 4 seconds\n")
        time.sleep(4)
        print("start read all axises offset value\n")
        self.mag()
        magTemp[3] = Mag[0]
        magTemp[4] = Mag[1]
        magTemp[5] = Mag[2]
        
        print("flip 10dof-imu device and keep it horizontal and it will read all axises offset value after 4 seconds\n")
        time.sleep(4)
        print("start read all axises offset value\n")
        self.mag()
        magTemp[6] = Mag[0]
        magTemp[7] = Mag[1]
        magTemp[8] = Mag[2]
        
        magOffset[0] = (magTemp[0]+magTemp[3])/2
        magOffset[1] = (magTemp[1]+magTemp[4])/2
        magOffset[2] = (magTemp[5]+magTemp[8])/2
        print(magOffset)
        '''
        magOffset[0] = 30.0
        magOffset[1] = 190.0
        magOffset[2] = -210
    
    def temp(self):
        th = Read_Byte(TEMP)
        tl = Read_Byte(TEMP + 1)
        tempRow = self.dataConv(tl,th)
        tempC = (tempRow / 340.0) + 36.53
        tempC = "%.2f" % tempC
        return {"TEMP": tempC}

    def imuAHRSupdata(self,gx, gy,gz,ax,ay,az,mx,my,mz):    
        norm=0.0
        hx = hy = hz = bx = bz = 0.0
        vx = vy = vz = wx = wy = wz = 0.0
        exInt = eyInt = ezInt = 0.0
        ex=ey=ez=0.0 
        halfT = 0.024
        global q0
        global q1
        global q2
        global q3
        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q2 = q2 * q2   
        q2q3 = q2 * q3
        q3q3 = q3 * q3          

        norm = float(1/math.sqrt(ax * ax + ay * ay + az * az))     
        ax = ax * norm
        ay = ay * norm
        az = az * norm

        norm = float(1/math.sqrt(mx * mx + my * my + mz * mz))      
        mx = mx * norm
        my = my * norm
        mz = mz * norm

        # compute reference direction of flux
        hx = 2 * mx * (0.5 - q2q2 - q3q3) + 2 * my * (q1q2 - q0q3) + 2 * mz * (q1q3 + q0q2)
        hy = 2 * mx * (q1q2 + q0q3) + 2 * my * (0.5 - q1q1 - q3q3) + 2 * mz * (q2q3 - q0q1)
        hz = 2 * mx * (q1q3 - q0q2) + 2 * my * (q2q3 + q0q1) + 2 * mz * (0.5 - q1q1 - q2q2)         
        bx = math.sqrt((hx * hx) + (hy * hy))
        bz = hz     

        # estimated direction of gravity and flux (v and w)
        vx = 2 * (q1q3 - q0q2)
        vy = 2 * (q0q1 + q2q3)
        vz = q0q0 - q1q1 - q2q2 + q3q3
        wx = 2 * bx * (0.5 - q2q2 - q3q3) + 2 * bz * (q1q3 - q0q2)
        wy = 2 * bx * (q1q2 - q0q3) + 2 * bz * (q0q1 + q2q3)
        wz = 2 * bx * (q0q2 + q1q3) + 2 * bz * (0.5 - q1q1 - q2q2)  

        # error is sum of cross product between reference direction of fields and direction measured by sensors
        ex = (ay * vz - az * vy) + (my * wz - mz * wy)
        ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
        ez = (ax * vy - ay * vx) + (mx * wy - my * wx)

        if (ex != 0.0 and ey != 0.0 and ez != 0.0):
            exInt = exInt + ex * Ki * halfT
            eyInt = eyInt + ey * Ki * halfT  
            ezInt = ezInt + ez * Ki * halfT

            gx = gx + Kp * ex + exInt
            gy = gy + Kp * ey + eyInt
            gz = gz + Kp * ez + ezInt

        q0 = q0 + (-q1 * gx - q2 * gy - q3 * gz) * halfT
        q1 = q1 + (q0 * gx + q2 * gz - q3 * gy) * halfT
        q2 = q2 + (q0 * gy - q1 * gz + q3 * gx) * halfT
        q3 = q3 + (q0 * gz + q1 * gy - q2 * gx) * halfT  

        norm = float(1/math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3))
        q0 = q0 * norm
        q1 = q1 * norm
        q2 = q2 * norm
        q3 = q3 * norm

    def getdata(self):
        Accel = self.accel()
        Gyro = self.gyro()
        Mag = self.mag()
        self.imuAHRSupdata(Gyro[0]/32.8*0.0175, Gyro[1]/32.8*0.0175,Gyro[2]/32.8*0.0175,
                Accel[0],Accel[1],Accel[2], Mag[0], Mag[0], Mag[2])
        pitch = math.asin(-2 * q1 * q3 + 2 * q0 * q2) * 57.3
        roll  = math.atan2(2 * q2 * q3 + 2 * q0 * q1, -2 * q1 * q1 - 2 * q2 * q2 + 1) * 57.3
        yaw   = math.atan2(-2 * q1 * q2 - 2 * q0 * q3, 2 * q2 * q2 + 2 * q3 * q3 - 1) * 57.3
        
        return [roll,pitch,yaw,Accel[0],Accel[1],Accel[2],Gyro[0],Gyro[1],Gyro[2],Mag[0],Mag[1],Mag[2]]

if __name__ == '__main__':
    sensor = MPU925x()
    icm = []
    try:
        while True:
            # print("while")
            # s.mag()
            time.sleep(0.5)
            icm = sensor.getdata()
            print("/-------------------------------------------------------------/")
            print("Roll = %.2f , Pitch = %.2f , Yaw = %.2f" %(icm[0],icm[1],icm[2]))
            print("Acceleration: X = %d, Y = %d, Z = %d" %(icm[3],icm[4],icm[5]))
            print("Gyroscope:     X = %d , Y = %d , Z = %d" %(icm[6],icm[7],icm[8]))
            print("Magnetic:      X = %d , Y = %d , Z = %d" %(icm[9],icm[10],icm[11]))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        # sensor.Disable()
        exit()