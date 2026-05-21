#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
#include <math.h>

#define DHTPIN 2
#define DHTTYPE DHT22

#define MQ135PIN A0
#define LDRPIN A1
#define LM35PIN A2

#define RL 10.0
#define R0 56.35
#define NUM_SAMPLES 15

// MQ135 correction constants (temp/humidity compensation)
#define CORA 0.00035
#define CORB 0.02718
#define CORC 1.39538
#define CORD 0.0018
#define CORE -0.003333333
#define CORF -0.001923077
#define CORG 1.130128205

DHT_Unified dht(DHTPIN, DHTTYPE);

uint32_t delayMS;
String receivedTimestamp = "";

// Minimal bridge payload format:
// temperature,humidity,co2,light
// (optional occupancy can be added as 5th field if needed later)

float readRS() {
  int adcValue = analogRead(MQ135PIN);
  if (adcValue <= 0 || adcValue >= 1023) {
    return 0.0;
  }

  // RS = RL * (1023 - ADC) / ADC. This avoids hard-coding Vref.
  return RL * (1023.0 - (float)adcValue) / (float)adcValue;
}

float getCO2FromRS(float rs) {
  if (rs <= 0.0) {
    return 0.0;
  }

  float ratio = rs / R0;
  if (ratio <= 0.0) {
    return 0.0;
  }

  return 400.0 * pow((3.6 / ratio), 1.5);
}

float getCorrectionFactor(float temperature, float humidity) {
  // MQ135 empirical correction from common library implementation.
  if (temperature < 20.0) {
    return CORA * temperature * temperature - CORB * temperature + CORC - (humidity - 33.0) * CORD;
  }
  return CORE * temperature + CORF * humidity + CORG;
}

void sortArray(float arr[], int size) {
  for (int i = 0; i < size - 1; i++) {
    for (int j = i + 1; j < size; j++) {
      if (arr[i] > arr[j]) {
        float temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }
}

float getStableCO2(float temperature, float humidity) {
  float readings[NUM_SAMPLES];

  for (int i = 0; i < NUM_SAMPLES; i++) {
    float rs = readRS();
    float corr = getCorrectionFactor(temperature, humidity);
    if (corr <= 0.0) {
      corr = 1.0;
    }

    float correctedRS = rs / corr;
    readings[i] = getCO2FromRS(correctedRS);
    delay(100);
  }

  sortArray(readings, NUM_SAMPLES);

  float sum = 0.0;
  for (int i = 2; i < NUM_SAMPLES - 2; i++) {
    sum += readings[i];
  }

  float avg = sum / (NUM_SAMPLES - 4);
  return avg;
}

void setup() {
  Serial.begin(9600);

  dht.begin();

  sensor_t sensor;
  dht.temperature().getSensor(&sensor);
  delayMS = sensor.min_delay / 1000;

  Serial.println("Arduino minimal bridge ready");
}

void loop() {
  if (Serial.available() > 0) {
    receivedTimestamp = Serial.readStringUntil('\n');
    receivedTimestamp.trim();

    if (receivedTimestamp.length() > 0) {
      delay(delayMS);

      // Read DHT temperature/humidity
      sensors_event_t event;
      dht.temperature().getEvent(&event);
      float temperature = event.temperature;

      dht.humidity().getEvent(&event);
      float humidity = event.relative_humidity;

      // Read light (LDR) and map to lux-like scale used in current setup
      int ldrValue = analogRead(LDRPIN);
      float lux = (ldrValue / 1023.0) * 1000.0;

      // Read CO2 (MQ135) using calibrated RS/R0 method with filtering
      // plus temperature/humidity compensation.
      float co2 = getStableCO2(temperature, humidity);

      // Fallbacks for failed sensor reads
      if (isnan(temperature)) {
        temperature = 0.0;
      }
      if (isnan(humidity)) {
        humidity = 0.0;
      }
      if (isnan(co2)) {
        co2 = 0.0;
      }
      if (isnan(lux)) {
        lux = 0.0;
      }

      // Send minimal payload required by simulation bridge
      Serial.print(temperature, 1);
      Serial.print(",");
      Serial.print(humidity, 1);
      Serial.print(",");
      Serial.print(co2, 1);
      Serial.print(",");
      Serial.println(lux, 1);
    }
  }
}
