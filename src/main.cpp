#include <Arduino.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "FS.h"
#include "LittleFS.h"
#include <ctype.h>
#include <U8g2lib.h>

#define FONT u8g2_font_wqy14_t_gb2312b
#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, 22, 21, U8X8_PIN_NONE);//CLK; DATA; RESET

#define TIME_STEPS 12

void read_file(String ,int ,int ,float *);// should start with /
uint64_t LSTM_neuron();
float sigmoid(float);

String entrada;
float dense_weights[TIME_STEPS + 1][1] = {0};//+ 1 por el h_(t-1)
float sesgo = 0;
float y = 0;


void setup() {
  pinMode(2,OUTPUT);
  Serial.begin(115200);
  u8g2.begin();
  u8g2.setFont(FONT);	// choose a suitable font
  if (!LittleFS.begin(true)) {
    Serial.println("Error to initiate LittleFS");
    return;
    //para cargar el sistema de archivos a la esp, darle en el icono de platformio, luego en PROJECT TASKS;
    //luego en Platform/Buils Filesystem Image y por ultimo Platform/Upload Filesystem Image
  }

  Serial.println("\n=== File system info ===");
  Serial.printf("Total space: %d\n",LittleFS.totalBytes());
  Serial.printf("Total space used: %d\n",LittleFS.usedBytes());
  Serial.println("========================\n");

  // read_file("/capa_embedding.txt",EMB_DIM,VOCAB_SIZE,(float *)emb);
  // read_file("/capa_dense.txt",1,MAX_ENT*EMB_DIM+1,(float *)dense_weights);
  sesgo = dense_weights[0][0]; //el primer dato es el sesgo
  delay(1000); 
}


void loop() {
  if (Serial.available()){ 
    //Ejecucion del modelo
    y = 0;
    y += sesgo;//salida capa densa
    y = sigmoid(y); //paso por función de activación
    Serial.printf("Probabilidad: %.8f\n",y);
    u8g2.drawStr(15,55,"Probabilidad:");
    u8g2.setCursor(95, 55); u8g2.print(String(y));
    u8g2.sendBuffer();
  }
}

void read_file(String path, int size_x, int size_y, float *matriz){
  File archivo = LittleFS.open(path, "r");
  String word;
  uint8_t indx = 0;
  if (!archivo) {
    Serial.println("¡No se pudo abrir el archivo!");
    return;
  }
  for(int i = 0; i < size_y; i++){
    word = archivo.readStringUntil('\n');
    for(int j = 0; j < size_x; j++){
      if(word.indexOf(",",indx) != -1){ //=-1 cuando no existe el caracter en la cadena
        *(matriz + i * size_x + j) = word.substring(indx,word.indexOf(',',indx)).toFloat();
        indx = word.indexOf(',',indx) + 1;
      }else{
        *(matriz + i * size_x + j) = word.substring(indx,word.indexOf('\n', indx)).toFloat();
        indx = 0;
      }
      //Serial.printf("%.6f,",*(matriz + i * size_x + j));
    }
    //Serial.println();
  }
}

uint64_t LSTM_neuron(){

}

float sigmoid(float z) {
  return 1.0 / (1.0 + exp(-z));
}