#!/usr/bin/env python3

import time
from datetime import datetime
import os, sys
import smbus2 as smbus
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SUPABASE_URL, SUPABASE_KEY, SAMPLE_RATE
from supabase import create_client
from python import ICM20948, MPU925x, BME280, LTR390, TSL2591, SGP40


class SensorService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")

        # Initialize Supabase client
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Initialize sensors
        self.bus = smbus.SMBus(1)
        self.icm_val_wia = 0xEA
        self.mpu_val_wia = 0x71
        self.icm_slave_address = 0x68

        self.device_id1 = self.bus.read_byte_data(int(self.icm_slave_address), 0x00)
        self.device_id2 = self.bus.read_byte_data(int(self.icm_slave_address), 0x75)

        if self.device_id1 == self.icm_val_wia:
            self.mpu = ICM20948.ICM20948()
            print("ICM20948 detected at I2C address 0x68")
        elif self.device_id2 == self.mpu_val_wia:
            self.mpu = MPU925x.MPU925x()
            print("MPU925x detected at I2C address 0x68")
        else:
            raise RuntimeError("No compatible IMU detected")

        self.bme280 = BME280.BME280()
        self.bme280.get_calib_param()
        self.light = TSL2591.TSL2591()
        self.uv = LTR390.LTR390()
        self.sgp = SGP40.SGP40()

        print("Sensors initialized successfully")

    def read_sensors(self):
        """
        Reads data from all sensors and returns them as a dictionary.
        """
        timestamp = datetime.utcnow()

        try:
            # Read BME280 (Temperature, Pressure, Humidity)
            bme_data = self.bme280.readData()
            pressure = round(bme_data[0], 2)
            temp = round(bme_data[1], 2)
            hum = round(bme_data[2], 2)

            # Read TSL2591 (Light)
            lux = round(self.light.Lux(), 2)

            # Read LTR390 (UV)
            uv_index = self.uv.UVS()

            # Read SGP40 (VOC)
            voc = round(self.sgp.raw(), 2)

            # Read IMU (Gyroscope, Accelerometer, Magnetometer)
            icm_data = self.mpu.getdata()
            roll, pitch, yaw = icm_data[:3]
            accel_x, accel_y, accel_z = icm_data[3:6]
            gyro_x, gyro_y, gyro_z = icm_data[6:9]
            mag_x, mag_y, mag_z = icm_data[9:12]

            return [
                {"timestamp": timestamp, "temperature": temp, "pressure": pressure, "humidity": hum,
                 "light": lux, "uv": uv_index, "voc": voc,
                 "accelerometer_x": accel_x, "accelerometer_y": accel_y, "accelerometer_z": accel_z,
                 "gyroscope_x": gyro_x, "gyroscope_y": gyro_y, "gyroscope_z": gyro_z,
                 "magnetometer_x": mag_x, "magnetometer_y": mag_y, "magnetometer_z": mag_z}
            ]
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return []

    def store_data(self, readings):
        """
        Stores the readings in Supabase.
        """
        if not readings:
            print("No data to store")
            return

        try:
            response = self.supabase.table("sensor_readings").insert(readings).execute()
            if response.get("status_code") != 201:
                print(f"Insert error: {response}")
            else:
                print("Data successfully inserted")
        except Exception as e:
            print(f"Error storing data: {e}")

    def run(self):
        """
        Main loop to collect and store sensor data at regular intervals.
        """
        print(f"Starting sensor service, sampling every {SAMPLE_RATE} seconds")
        while True:
            readings = self.read_sensors()
            self.store_data(readings)
            time.sleep(SAMPLE_RATE)


if __name__ == "__main__":
    service = SensorService()
    service.run()
