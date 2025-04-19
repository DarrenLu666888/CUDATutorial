#include "stdio.h"
int main(){
  char* str = "Hello, World!";
  printf("%c\n", str[0]);

  printf("%x\n", str[0]);
  printf("%x\n", str[1]);

  short* p = (short*)str;
  printf("%x\n", p[0]);
}