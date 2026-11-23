#ifndef SEIYUU_FACEID_H
#define SEIYUU_FACEID_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct faceid_engine faceid_engine;
/* All returned strings are allocated by the core and freed by faceid_free.
 * Calls on one engine are serialized by the caller; destroy after calls finish.
 * No C++ exception crosses this boundary. Success returns 0, failure 1.
 * Error strings are JSON {"error":{"code":...,"message":...}}.
 */
int faceid_create(const char *model_json_path, faceid_engine **engine, char **error);
void faceid_destroy(faceid_engine *engine);
void faceid_free(char *text);
int faceid_model(faceid_engine *engine, char **result);
/* mode: 0 recognize, 1 detect, 2 classify an already aligned 112x112 BGR crop,
 * 3 detect boxes/scores only. In mode 3, threshold sets detector confidence
 * for this call, with 0 selecting the model configuration's detector threshold.
 * mode 2 returns {"features":[1024 floats],"probabilities":[N floats],"candidates":[...]}
 * Other modes return {"width":W,"height":H,"faces":[API-compatible face objects]}.
 * top_k must be 1..class_count; threshold finite in [0,1].
 */
int faceid_process_bgr(faceid_engine *engine, const unsigned char *pixels,
                       size_t length, int width, int height, int mode,
                       int top_k, double threshold, char **result);
/* Decode JPEG/PNG/WebP with OpenCV, then run the same pipeline.
 * HTTP layer checks encoding, animation, byte and pixel bounds first.
 */
int faceid_process_encoded(faceid_engine *engine, const unsigned char *bytes,
                           size_t length, int mode, int top_k,
                           double threshold, char **result);
#ifdef __cplusplus
}
#endif
#endif
