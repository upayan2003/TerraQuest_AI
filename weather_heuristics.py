import requests
from datetime import datetime, timedelta
import time

class Weather:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = f"https://api.weatherapi.com/v1/history.json"
    
    def get_unix_timestamp(self, days_ago):
        dt = datetime.now() - timedelta(days=days_ago)
        return int(time.mktime(dt.timetuple()))
    
    def fetch_weather_data(self, lat, lon, days=2):
        rainfall = 0.0
        snowfall = 0.0
        temp_avg = 0.0

        for day in range(1, days + 1):
            date_str = (datetime.now() - timedelta(days=day)).strftime('%Y-%m-%d')
            params = {
                "key": self.api_key,
                "q": f"{lat},{lon}",
                "dt": date_str
            }

            response = requests.get(self.base_url, params=params)

            if response.status_code != 200:
                print(f"Error fetching data for {date_str}: {response.text}")
                continue

            data = response.json()
            try:
                day_data = data["forecast"]["forecastday"][0]["day"]
                daily_temp = day_data.get("avgtemp_c", 0.0)
                daily_rain = day_data.get("totalprecip_mm", 0.0)
                daily_snow = day_data.get("totalsnow_cm", 0.0) * 10  # convert cm to mm

                temp_avg += daily_temp
                rainfall += daily_rain
                snowfall += daily_snow

            except (IndexError, KeyError) as e:
                print(f"Missing data for {date_str}: {e}")

        temp_avg /= days
        return rainfall, snowfall, temp_avg

    def assess_terrain_condition(self, lat, lon):
        rain, snow, temp = self.fetch_weather_data(lat, lon)
        location_weather = f"Rain: {rain:.2f} mm | Snow: {snow:.2f} mm | Avg Temp: {temp:.2f}°C"

        if rain > 5 and temp > 3:
            condition = "Mud Prone"
        elif snow > 3 and temp <= 0:
            condition = "Snow Prone"
        else:
            condition = "Normal"
        
        return condition, location_weather

# Example usage
if __name__ == "__main__":
    with open("WeatherAPI_Key.txt", "r") as f:
        API_KEY = f.read().strip()

    checker = Weather(API_KEY)

    lat, lon = float(input("Enter latitude: ")), float(input("Enter longitude: "))

    terrain_status = checker.assess_terrain_condition(lat, lon)
    print(f"Predicted Terrain Condition: {terrain_status}")

# git commit check
