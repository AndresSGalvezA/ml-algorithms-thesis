#include "DEV_Config.h"
#include "Waveshare_AS7341.h"
#include "EloquentMLPortRandomForest.h"

Eloquent::ML::Port::RandomForest forest;

const int greenBananaPin = 2;
const int lightBananaPin = 3;
const int yellowBananaPin = 4;
const int blackBananaPin = 5;
const int btnPin = 6;

void setup() {
  pinMode(btnPin, INPUT);
  pinMode(blackBananaPin, OUTPUT);
  pinMode(greenBananaPin, OUTPUT);
  pinMode(lightBananaPin, OUTPUT);
  pinMode(yellowBananaPin, OUTPUT);

  DEV_ModuleInit();
  Serial.print("IIC ready! Now start initializing AS7341!\r\n");
  AS7341_Init(eSpm);
  AS7341_ATIME_config(100);
  AS7341_ASTEP_config(999);
  AS7341_AGAIN_config(6);
  AS7341_EnableLED(true);

  Serial.begin(9600);

}

void loop() {
  int btnState = digitalRead(btnPin);

  if (btnState == HIGH) {
    AS7341_ControlLed(true,10);
    sModeOneData_t data1;
    sModeTwoData_t data2;
    AS7341_startMeasure(eF1F4ClearNIR);
    data1 = AS7341_ReadSpectralDataOne();
    Serial.print("channel1(405-425nm):");
    Serial.println(data1.channel1);
    Serial.print("channel2(435-455nm):");
    Serial.println(data1.channel2);
    Serial.print("channel3(470-490nm):");
    Serial.println(data1.channel3);
    Serial.print("channel4(505-525nm):");   
    Serial.println(data1.channel4);
    AS7341_startMeasure(eF5F8ClearNIR);
    data2 =AS7341_ReadSpectralDataTwo();
    Serial.print("channel5(545-565nm):");
    Serial.println(data2.channel5);
    Serial.print("channel6(580-600nm):");
    Serial.println(data2.channel6);
    Serial.print("channel7(620-640nm):");
    Serial.println(data2.channel7);
    Serial.print("channel8(670-690nm):");
    Serial.println(data2.channel8);
    Serial.print("Clear:");
    Serial.println(data2.CLEAR);
    Serial.print("NIR:");
    Serial.println(data2.NIR);
    Serial.print("--------------------------\r\n");
    AS7341_ControlLed(false,10);

    float x[9] = {data1.channel1, data1.channel2, data1.channel3, data1.channel4, data2.channel5, data2.channel6, data2.channel7, data2.channel8, data2.NIR};

    int prediction = forest.predict(x);
    
    Serial.println(prediction);

    digitalWrite(blackBananaPin, LOW);
    digitalWrite(greenBananaPin, LOW);
    digitalWrite(lightBananaPin, LOW);
    digitalWrite(yellowBananaPin, LOW);

    switch (prediction) {
      case 0:
        digitalWrite(blackBananaPin, HIGH);
        break;
      case 1:
        digitalWrite(greenBananaPin, HIGH);
        break;
      case 2:
        digitalWrite(lightBananaPin, HIGH);
        break;
      case 3:
        digitalWrite(yellowBananaPin, HIGH);
        break;
    }
  }
}