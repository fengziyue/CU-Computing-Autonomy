/*  Nick Sweeting 2013/10/09
    References vs Values in C++
    MIT License

    Takes input and removes puctuation and spaces, using two different methods.
    It is referred from: https://github.com/pirate/Cpp-Data-Structures/
*/

#include <stdlib.h>
#include <iostream>
using namespace std;

// returns true if input character q is puctuation, else false
bool ispunctuation(char q) {
    // complete the functions here ...
}

char* modifyAndCopy(char *raw_input) {
    // complete the functions here ...
}

char* modifyInPlace(char *raw_input) {
    // complete the functions here ...
}

int main() {
    // user input
    char raw_input[80] = {0};
    cout << "Please input something with punctuation in it: ";
    cin.getline(raw_input,80);

    cout << "Modify and Copy: " << endl;
    cout << "Original: " << raw_input << endl;
    cout << "Modified: " << modifyAndCopy(raw_input) << endl << endl;

    cout << "Modify in Place: " << endl;
    cout << "Original: " << raw_input << endl;
    cout << "Modified: " << modifyInPlace(raw_input) << endl;
}
