const byte Moteur_1_avant = 8;
const byte Moteur_1_arr = 7;
const byte Moteur_2_avant = 2;
const byte Moteur_2_arr = 4;
const byte en1 = 3;/*-sur le pont en H - Moteur1*/
const byte en2 = 9;/*-sur le pont en H - Moteur2*/

/*Liaison série*/
#include "SoftwareSerial.h"
//SoftwareSerial mySerial(5, 4); // RX | TX

/* Bibliothéque pour les servos */
#include <Servo.h>
Servo servo_v;/*servo qui gére le déplacement vertical*/
Servo servo_h;/*servo qui gére le déplacement horizontal*/


char command;//Commande pour la liaison série
int angle_h;//Angle du servo sui gère l'horizontale
int angle_v;

void setup() {

Serial.begin(9600);//57600
Serial.println("Connexion série OK");

servo_v.attach(10);//Commande pwm du servo horizontale sur la broche 10
servo_h.attach(11);

/*initialisation positionnement caméra au centre */
angle_h=90;
angle_v=90;
servo_v.write(angle_v);
servo_h.write(angle_h);

/*Declaration moteurs*/
pinMode(Moteur_1_avant,OUTPUT);//Moteur 1
pinMode(Moteur_1_arr,OUTPUT);//Moteur 1
pinMode(Moteur_2_avant,OUTPUT);//Moteur 2
pinMode(Moteur_2_arr,OUTPUT);//Moteur 2

pinMode(en1,OUTPUT);//Enable motor 1
pinMode(en2,OUTPUT);//Enable motor 2
}

void loop() {

if (Serial.available())
   {
    command=(Serial.read());
    switch (command) {
        
        case 'a': // Robot Tank Forward
        analogWrite(en1,255);
        analogWrite(en2,255);
        digitalWrite(Moteur_1_avant, HIGH);
        digitalWrite(Moteur_2_avant, HIGH);
        digitalWrite(Moteur_1_arr, LOW);
        digitalWrite(Moteur_2_arr, LOW);
        Serial.println("Robot Tank Forward");
        break;

        case 'r': // Robot Tank Backward
        analogWrite(en1,255);
        analogWrite(en2,255);
        digitalWrite(Moteur_1_avant, LOW);
        digitalWrite(Moteur_2_avant, LOW);
        digitalWrite(Moteur_1_arr, HIGH);
        digitalWrite(Moteur_2_arr, HIGH);
        Serial.println("Robot Tank Backward");
        break;

        case 'g': // Robot Tank Left
        analogWrite(en1,255);
        analogWrite(en2,255);
        digitalWrite(Moteur_1_avant, HIGH);
        digitalWrite(Moteur_2_avant, LOW);
        digitalWrite(Moteur_1_arr, LOW);
        digitalWrite(Moteur_2_arr, HIGH);
        Serial.println("Robot Tank Left");
        break;

        case 'd': // Robot Tank Right
        analogWrite(en1,255);
        analogWrite(en2,255);
        digitalWrite(Moteur_1_avant, LOW);
        digitalWrite(Moteur_2_avant, HIGH);
        digitalWrite(Moteur_1_arr, HIGH);
        digitalWrite(Moteur_2_arr, LOW);
        Serial.println("Robot Tank Right");
        break;

        case 'h': //Cam horizontale vers la gauche
        /*servo_v.write(angle_v);*/
        
        if (angle_h <= 165)
          {
          angle_h=angle_h+15;
          servo_h.write(angle_h);
          Serial.println(angle_h);
          }
        break;

        case 'j': //Cam horizontale vers la droite
        if (angle_h >= 15)
          {
          angle_h=angle_h-15;
          servo_h.write(angle_h);
          Serial.println(angle_h);     
          }
        break;

        case 'v': //Cam horizontale vers le bas    
        if (angle_v <= 165)
          {
          angle_v=angle_v+15;
          servo_v.write(angle_v);
          Serial.println(angle_v);
          }
        break;

        case 'f': //Cam horizontale vers le haut     
        if (angle_v >= 15)
          {
          angle_v=angle_v-15;
          servo_v.write(angle_v);
          Serial.println(angle_v);
          }
        break;
        default: //Stop The Robot Tank 
        analogWrite(en1,0);
        analogWrite(en2,0);
        //digitalWrite(Moteur_2_avant, LOW);
        //digitalWrite(Moteur_2_arr, LOW);
        Serial.println("Robot Tank Stop");
        }
   }
        else {
        Serial.println("Attente d'une commande");
        }
        
  }

