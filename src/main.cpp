#include <Arduino.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "FS.h"
#include "LittleFS.h"
#include <ctype.h>
#include <U8g2lib.h>

#include "esp_heap_caps.h"

#define FONT u8g2_font_wqy14_t_gb2312b
#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, 22, 21, U8X8_PIN_NONE);//CLK; DATA; RESET

//#define TIME_STEPS 12
#define VOCAB_SIZE 1000
#define MAX_ENT 15
#define EMB_DIM 16
#define LSTM_UNITS 50
#define DENSE_UNITS 4

void read_file(String ,int ,int ,float *);
uint64_t hash1(const char *);
uint64_t hash2(uint64_t);
void padded_right(uint16_t *, char *, int);
void onehot(float *, uint16_t *);
void LSTM_neuron(float *, float *, float *, float *, float *, float *);
void dense_neuron(float *, float *, float *, float *, const char *);
void multMatriz(float *, uint16_t, uint16_t, float *, uint16_t, float *);
float sigmoid(float);

String entrada;
uint16_t ent_encoded[MAX_ENT] = {0};
float output_emb[MAX_ENT][EMB_DIM] = {0};
float ent_encod_onehot[MAX_ENT][VOCAB_SIZE] = {0};
float h_t_LSTM[LSTM_UNITS][4*LSTM_UNITS] = {0};//no cabe en malloc
/*float emb[VOCAB_SIZE][EMB_DIM] = {0};float capa_LSTM[EMB_DIM][4*LSTM_UNITS] = {0};
float h_t_LSTM[LSTM_UNITS][4*LSTM_UNITS] = {0};float sesgos_LSTM[1][LSTM_UNITS] = {0};
float capa_densa[LSTM_UNITS][DENSE_UNITS] = {0};float sesgos_dense[1][DENSE_UNITS] = {0};*/
//si al imprimir la direccion de la variable devuelve 0x000 o 0xFFF, malloc falló

float C_t[1][LSTM_UNITS] = {0}; float h_t[1][LSTM_UNITS] = {0};
float yout[1][4] = {0};
//recorrer matrices con punteros: *(matriz + i * columnas + j) = matriz[i][j]

void setup() {
  pinMode(2,OUTPUT);
  Serial.begin(115200);
  u8g2.begin();
  u8g2.setFont(FONT);	// choose a suitable font
  if (!LittleFS.begin(true)) {
    Serial.println("Error to initiate LittleFS");
    return;
    //para cargar el sistema de archivos a la esp, darle en el icono de platformio, luego en la ventana PROJECT TASKS;
    //luego en "Platform/Build Filesystem Image" y por ultimo "Platform/Upload Filesystem Image"
  }

  Serial.println("\n=== File system info ===");
  Serial.printf("Total space: %d\n",LittleFS.totalBytes());
  Serial.printf("Total space used: %d\n",LittleFS.usedBytes());
  Serial.println("========================\n");
  read_file("/h_t_lstm.bin",LSTM_UNITS,4*LSTM_UNITS,(float *)(h_t_LSTM));
  delay(1000); 
}

void loop() {
  if (Serial.available()){ //ingresar frase
    entrada = Serial.readStringUntil('\n').c_str();  // Lee todo el String recibido
    entrada.toLowerCase();
    Serial.print("\nRecibido: "); Serial.println(entrada);

    padded_right((uint16_t *)ent_encoded, (char *)entrada.c_str(), MAX_ENT);
    onehot((float *)ent_encod_onehot, (uint16_t *)ent_encoded);
    
    float* emb = (float*)malloc(VOCAB_SIZE * EMB_DIM * sizeof(float));//leer directamente la matriz de la flash
    read_file("/capa_embedding.bin",VOCAB_SIZE,EMB_DIM,emb);// should start with /
    multMatriz((float *)ent_encod_onehot,VOCAB_SIZE, MAX_ENT, emb, EMB_DIM, (float *)output_emb);//capa embedding
    free(emb);
    for(int i = 0; i < MAX_ENT; i++){
      for(int j = 0; j < EMB_DIM; j++){
        Serial.printf("%.5f ", output_emb[i][j]);
      }Serial.println(); 
    }
    Serial.println(); 
    float* capa_LSTM = (float*)malloc(EMB_DIM * 4 * LSTM_UNITS * sizeof(float));
    float* sesgos_LSTM = (float*)malloc(4 * LSTM_UNITS * sizeof(float));
    read_file("/lstm.bin",EMB_DIM,4*LSTM_UNITS,capa_LSTM);
    read_file("/sesgos_lstm.bin",1,4*LSTM_UNITS,sesgos_LSTM);
    LSTM_neuron((float *)(capa_LSTM),//Capa LSTM
                (float *)(h_t_LSTM), (float *)(sesgos_LSTM),
                (float *)(output_emb), (float *)(h_t), (float *)(C_t));
    free(capa_LSTM); free(sesgos_LSTM);
    
    float* sesgos_dense = (float*)malloc(1 * DENSE_UNITS * sizeof(float));
    float* capa_densa = (float*)malloc(LSTM_UNITS * DENSE_UNITS * sizeof(float));
    read_file("/sesgos_densa.bin",1,DENSE_UNITS,sesgos_dense);
    read_file("/capa_densa.bin",LSTM_UNITS,DENSE_UNITS,capa_densa);
    for(int j = 0; j < DENSE_UNITS; j++){//Capa densa
      dense_neuron((float *)(capa_densa + j),(float *)(sesgos_dense+j),(float *)(h_t),&yout[0][j],"sigmoid");
    }
    free(capa_densa); free(sesgos_dense);

    for(int j = 0; j < DENSE_UNITS; j++){
      Serial.printf("yout[%d] = %.5f ", j, yout[0][j]);
    }
    Serial.println(); 
    u8g2.clearBuffer();
    u8g2.drawStr(15,25,"feliz:");u8g2.drawStr(15,35,"triste:");
    u8g2.drawStr(15,45,"aburi:");u8g2.drawStr(15,55,"brav:");
    u8g2.setCursor(55, 25); u8g2.print(String(yout[0][0],6));
    u8g2.setCursor(55, 35); u8g2.print(String(yout[0][1],6));
    u8g2.setCursor(55, 45); u8g2.print(String(yout[0][2],6));
    u8g2.setCursor(55, 55); u8g2.print(String(yout[0][3],6));
    u8g2.sendBuffer();
  }
}

void read_file(String path, int size_x, int size_y, float *matriz){
  File archivo = LittleFS.open(path, "rb");  // Abre en modo binario (rb = read binary)
  if (!archivo) {
      Serial.println("¡No se pudo abrir el archivo!");
      return;
  }
  size_t total_elementos = size_x * size_y;
  size_t bytes_leidos = archivo.read((uint8_t*)matriz, total_elementos * sizeof(float));
  archivo.close();
  if (bytes_leidos != total_elementos * sizeof(float)) {
      Serial.printf("Error: Se leyeron %d bytes en lugar de %d\n", bytes_leidos, total_elementos * sizeof(float));
  } else {
      Serial.println("Archivo binario leído correctamente.");
  }
  /*for(int i = 0; i < size_y; i++){
    for(int j = 0; j < size_x; j++){
      Serial.printf("%.6f,",*(matriz + i * size_x + j));
    }
    Serial.println();
  }*/
}

uint64_t hash1(const char *str){
  uint64_t hash = 7919;
  int c;
  int i = 0;
  while ((c = *(str +i))) {
    if ((c != '\0') && (c != '\n') && (c != '\r')){
      hash = ((hash << 5) + hash) + c;
    } 
    i+=1;//fuera del if para asegurar que sale del bucle
  }
  return hash;
}

uint64_t hash2(uint64_t hash){
  return hash * 65599; //otro numero primo grande se necesita un int de 64bits, si no trunca el resultado
}

void padded_right(uint16_t *in_encoded, char *input, int max_ent){
  char *token[MAX_ENT]; uint8_t i = 0;
  while ((token[i] = strtok_r(input, " ", &input))) {  
    printf("token: %s\n", token[i]);
    i++;
  }
  for(uint8_t j = 0; j < MAX_ENT; j++){
    *(in_encoded + j) = (j < (MAX_ENT-i)) ? 0 : hash2(hash1(token[j - (MAX_ENT - i)])) % VOCAB_SIZE; //se hace el padded a la derecha
    printf(" %d ", *(in_encoded + j));
  }
  printf("= padded\n");
}

void onehot(float *one_hot, uint16_t *in_encoded){
  memset(one_hot, 0, MAX_ENT*VOCAB_SIZE*sizeof(one_hot));//reiniciar la matriz
  for(int i = 0; i < MAX_ENT; i++){
    //ent_encod_onehot[i][ent_encoded[i]] = 1;
    *(one_hot + i*VOCAB_SIZE + *(in_encoded + i)) = 1;
  }
}

void LSTM_neuron(float *weights, float *h_t_weights, float *sesgos, float *x, float *h, float *C){
  float f_t = 0;   float i_t = 0;
  float o_t = 0;   float _C_t = 0;//~C_t
  float z1[1][4*LSTM_UNITS] = {0};
  float z2[1][4*LSTM_UNITS] = {0};
  // Inicializar C y h a cero al comenzar
  for(int j = 0; j < LSTM_UNITS; j++){
    *(C + j) = 0.0;
    *(h + j) = 0.0;
  }
  //Ejecucion del modelo
  for(int i = 0; i < MAX_ENT; i++){
    multMatriz((x + i*EMB_DIM),EMB_DIM,1,weights,4*LSTM_UNITS,(float *)z1);
    multMatriz(h,LSTM_UNITS,1,h_t_weights,4*LSTM_UNITS,(float *)z2);
    for(int j = 0; j < 4*LSTM_UNITS; j++){
      z2[0][j] += z1[0][j] + *(sesgos + j);//output compuertas LSTM
    }
    for(int j = 0; j < LSTM_UNITS; j++){
      i_t = sigmoid(z2[0][j]);//calculo de i_t
      f_t = sigmoid(z2[0][LSTM_UNITS + j]);//calculo de f_t
      _C_t = tanh(z2[0][2*LSTM_UNITS + j]);//calculo de _C_t
      o_t = sigmoid(z2[0][3*LSTM_UNITS + j]);//calculo de o_t
      *(C + j) = f_t*(*(C + j)) + i_t*_C_t;
      *(h + j) = tanh(*(C + j))*o_t;
    }
  }
}

void dense_neuron(float *weights, float *sesgo, float *intput, float *result, const char *func_act){
  *result = *sesgo;//reiniciar, se suma el sesgo
  for(int i = 0; i < LSTM_UNITS; i++){
    *result += *(weights + i*DENSE_UNITS) * *(intput + i);//*DENSE_UNITS es para acceder de columna en columna de la capa densa
  }
  if(strcmp(func_act, "sigmoid") == 0){
    *result = sigmoid(*result);//se aplica la funcion de activacion
  }else if(strcmp(func_act, "tanh") == 0){
    *result = tanh(*result);  //se aplica la funcion de activacion
  }
}

void multMatriz(float *mat1, uint16_t sizex1_y2, uint16_t sizey1, float *mat2, uint16_t sizex2, float *result) {
  for (int i = 0; i < sizey1; i++) {
    for (int j = 0; j < sizex2; j++) {
      *(result + i*sizex2 + j) = 0; //reiniciar la matriz
      for (int k = 0; k < sizex1_y2; k++) {//(addr_matriz + fila*no_column + columna)
        *(result + i*sizex2 + j) += *(mat1 + i*sizex1_y2 + k) * *(mat2 + k*sizex2 + j);  // Producto fila-columna
      }
    }
  }
}

float sigmoid(float z) {
  return 1.0 / (1.0 + exp(-z));  
}