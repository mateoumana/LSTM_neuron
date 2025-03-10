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

void read_file(String ,int ,int ,float *);
float LSTM_neuron(float *, float *, float *, float *, float *, float *, float *, float *, float *);
void dense_neuron(float *, float *, float *, const char *);
float sigmoid(float *);

//Esperando un vector de pesos de la siguiente forma: sesgo+pesos
float dense_f_weights[1][TIME_STEPS + 1] = {0};//+ 1 por el sesgo
float dense_i_weights[1][TIME_STEPS + 1] = {0};
float dense_C_weights[1][TIME_STEPS + 1] = {0};
float dense_o_weights[1][TIME_STEPS + 1] = {0};
float C_t = 0; float C_t_1 = 0;
float h_t = 0; float h_t_1 = 0;
float x_t[1][TIME_STEPS] = {0};//entradas
//recorrer matrices con punteros: *(matriz + i * columnas + j) = matriz[i][j]

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
  
  read_file("/weights_dense_f.txt",1,TIME_STEPS+1,(float *)(dense_f_weights));// should start with /
  read_file("/weights_dense_i.txt",1,TIME_STEPS+1,(float *)(dense_i_weights));
  read_file("/weights_dense_C.txt",1,TIME_STEPS+1,(float *)(dense_C_weights));
  read_file("/weights_dense_o.txt",1,TIME_STEPS+1,(float *)(dense_o_weights));
  delay(1000); 
}


void loop() {
  if (Serial.available()){ //presionar enter para ejecutar la neurona
    LSTM_neuron((float *)(dense_f_weights),//se pasa todo el vector y en la funcion se separa el sesgo
                (float *)(dense_C_weights),
                (float *)(dense_i_weights),
                (float *)(dense_o_weights),
                (float *)(x_t), &h_t, &h_t_1, &C_t, &C_t_1);
    Serial.printf("h_t: %.8f\n",h_t);
    u8g2.drawStr(15,55,"h_t:");
    u8g2.setCursor(55, 55); u8g2.print(String(h_t));
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

float LSTM_neuron(float *f_weights, float *C_weights, float *i_weights, float *o_weights, 
                  float *x, float *h, float *h_1, float *C, float *C_1){
  float f_t = 0;   float i_t = 0;
  float o_t = 0;   float _C_t = 0;//~C_t
  //Ejecucion del modelo
  *(x + 0) = *h_1;//se añade h_t_1 a la entrada general, al inicio
  dense_neuron((float *)(dense_f_weights), (float *)x, &f_t, "sigmoid"); //calculo de f_t
  dense_neuron((float *)(dense_C_weights), (float *)x, &_C_t, "sigmoid"); //calculo de i_t
  dense_neuron((float *)(dense_i_weights), (float *)x, &i_t, "tanh");    //calculo de _C_t
  dense_neuron((float *)(dense_o_weights), (float *)x, &o_t, "sigmoid"); //calculo de o_t
  *C = f_t*(*C_1) + i_t*_C_t;
  *h = tanh(*C)*o_t;
  *h_1 = *h;//guardar el valor anterior
  *C_1 = *C;
}

void dense_neuron(float *weights, float *intput, float *result, const char *func_act){
  for(int i = 0; i < TIME_STEPS; i++){
    *result += *(weights + i + 1) * *(intput + i); //weights + 1 para evadir el sesgo
  }
  *result += *(weights + 0);//se suma el sesgo (primera posición)
  if(strcmp(func_act, "sigmoid") == 0){
    *result = sigmoid(result);//se aplica la funcion de activacion
  }else if(strcmp(func_act, "tanh") == 0){
    *result = tanh(*result);  //se aplica la funcion de activacion
  }
}

float sigmoid(float *z) {
  return 1.0 / (1.0 + exp(-*z));
}