#!/usr/bin/env python3

import time
from datetime import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SUPABASE_URL, SUPABASE_KEY, SAMPLE_RATE
from supabase import create_client
from typing import List, Dict, Any


class MockSensor:
    def read_accelerometer(self): return (0, 0, 0)
    def read_gyroscope(self): return (0, 0, 0)
    def read_magnetometer(self): return (0, 0, 0)
    def read_temperature(self): return 20
    def read_pressure(self): return 1000
    def read_humidity(self): return 50
    def read_uv(self): return 0
    def read_lux(self): return 500
    def read_voc(self): return 100


class SensorService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.mock = MockSensor()

    def read_sensors(self) -> List[Dict[str, Any]]:
        """
        Reads mock sensor data and prepares it for insertion.
        """
        timestamp = datetime.utcnow()  # Use datetime object directly
        return [
            {
                "sensor_id": "icm20948",
                "timestamp": timestamp,
                "temperature": None,
                "humidity": None,
                "pressure": None,
                "light": None,
                "uv": None,
                "voc": None,
                "accelerometer_x": ax,
                "accelerometer_y": ay,
                "accelerometer_z": az,
                "gyroscope_x": gx,
                "gyroscope_y": gy,
                "gyroscope_z": gz,
                "magnetometer_x": mx,
                "magnetometer_y": my,
                "magnetometer_z": mz,
            }
            for ax, ay, az, gx, gy, gz, mx, my, mz in [
                (
                    self.mock.read_accelerometer(),
                    self.mock.read_gyroscope(),
                    self.mock.read_magnetometer(),
                )
            ]
        ] + [
            {
                "sensor_id": "bme280",
                "timestamp": timestamp,
                "temperature": self.mock.read_temperature(),
                "humidity": self.mock.read_humidity(),
                "pressure": self.mock.read_pressure(),
                "light": None,
                "uv": None,
                "voc": None,
                "accelerometer_x": None,
                "accelerometer_y": None,
                "accelerometer_z": None,
                "gyroscope_x": None,
                "gyroscope_y": None,
                "gyroscope_z": None,
                "magnetometer_x": None,
                "magnetometer_y": None,
                "magnetometer_z": None,
            },
            {
                "sensor_id": "ltr390",
                "timestamp": timestamp,
                "temperature": None,
                "humidity": None,
                "pressure": None,
                "light": None,
                "uv": self.mock.read_uv(),
                "voc": None,
                "accelerometer_x": None,
                "accelerometer_y": None,
                "accelerometer_z": None,
                "gyroscope_x": None,
                "gyroscope_y": None,
                "gyroscope_z": None,
                "magnetometer_x": None,
                "magnetometer_y": None,
                "magnetometer_z": None,
            },
            {
                "sensor_id": "tsl25911",
                "timestamp": timestamp,
                "temperature": None,
                "humidity": None,
                "pressure": None,
                "light": self.mock.read_lux(),
                "uv": None,
                "voc": None,
                "accelerometer_x": None,
                "accelerometer_y": None,
                "accelerometer_z": None,
                "gyroscope_x": None,
                "gyroscope_y": None,
                "gyroscope_z": None,
                "magnetometer_x": None,
                "magnetometer_y": None,
                "magnetometer_z": None,
            },
            {
                "sensor_id": "sgp40",
                "timestamp": timestamp,
                "temperature": None,
                "humidity": None,
                "pressure": None,
                "light": None,
                "uv": None,
                "voc": self.mock.read_voc(),
                "accelerometer_x": None,
                "accelerometer_y": None,
                "accelerometer_z": None,
                "gyroscope_x": None,
                "gyroscope_y": None,
                "gyroscope_z": None,
                "magnetometer_x": None,
                "magnetometer_y": None,
                "magnetometer_z": None,
            },
        ]

    def store_data(self, readings: List[Dict[str, Any]]) -> None:
        """
        Inserts validated sensor data into the database.
        """
        try:
            if readings:
                response = self.supabase.table("sensor_readings").insert(readings).execute()
                print(f"Insert Response: {response}")
            else:
                print("No valid sensor data to store.")
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
