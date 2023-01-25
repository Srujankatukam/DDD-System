#define BUZZER_PIN 32
#define BUZZER_CHANNEL 0
int data;
void setup(){
  Serial.begin(115200);
  pinMode(13,OUTPUT);
  }
void loop(){
  data=Serial.read();
  if(data=='1')
  {
  ledcAttachPin(BUZZER_PIN, BUZZER_CHANNEL);
  ledcWriteTone(BUZZER_CHANNEL,1500);
  delay(8000);
  digitalWrite(13,1);
  ledcDetachPin(BUZZER_PIN);
  exit(0);
  
  }
  if(data=='2'){
  ledcDetachPin(BUZZER_PIN);
  digitalWrite(13,0);
  exit(0);
 
  }
  
}

