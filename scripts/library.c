
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stdlib.h>

int compare (const void * a, const void * b) {
    size_t fa = strlen((const char *)a);
    size_t fb = strlen((const char *)b);
    return (fa < fb) - (fa > fb);
}

char **convert_tokens_to_bpes(int og_string_len, char *original_string, char **vocab_list, int vocab_list_len){

    char **token_list = malloc(100*sizeof(char*));
    for(int i = 0; i < 100; i++)
        token_list[i] = (char*) calloc(100, sizeof(char ));

    char working_string[og_string_len];
    strcpy(working_string, original_string);
    int working_string_len = og_string_len;
    char *tokens_in_string[50];
    int tokens_in_string_len = 0;

    for(int i = 0; i < vocab_list_len; i++){

        if(strstr(original_string, vocab_list[i]) != NULL){
            tokens_in_string[tokens_in_string_len] = vocab_list[i];
            tokens_in_string_len++;
        }
    }
    char new_tokens_in_string[tokens_in_string_len][50];
    for (int i = 0; i<tokens_in_string_len; i++) {
        strcpy(new_tokens_in_string[i], tokens_in_string[i]);
    }

    qsort(new_tokens_in_string, tokens_in_string_len, 50, compare);
    int token_list_len = 0;
    while(working_string_len != 0){
        int have_candidate = 0;
        char candidate[100];
        for(int i = 0; i < tokens_in_string_len; i++){
            char sub_working_string[strlen(new_tokens_in_string[i])];
            memcpy(sub_working_string, &working_string[0], strlen(new_tokens_in_string[i]) );
            sub_working_string[strlen(new_tokens_in_string[i])] = '\0';

            if(strcmp(sub_working_string, new_tokens_in_string[i]) == 0){
                strcpy(candidate, new_tokens_in_string[i]);
                have_candidate = 1;
                break;
            }
        }


        if(have_candidate == 1){
            strcpy(token_list[token_list_len], candidate);
            token_list_len++;

            memcpy(working_string, &working_string[strlen(candidate)], strlen(working_string) );
            working_string[strlen(working_string)] = '\0';
            working_string_len = strlen(working_string);
        }
    }
    return token_list;
}


