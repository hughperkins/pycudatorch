#pragma once

#include <iostream>
#include <string>

class THFloatTensor;
class THFloatStorage;
class THCState;
struct lua_State;
class THCudaTensor;

THCState *getState(lua_State *L);
THCudaTensor *popCudaTensor(lua_State *L);
void pushCudaTensor(THCState *state, lua_State *L, THCudaTensor *tensor);

