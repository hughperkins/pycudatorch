extern "C" {
    #include "lua.h"
    #include "lauxlib.h"
    #include "lualib.h"
}

#ifndef _WIN32
    #include <dlfcn.h>
#endif

#include <iostream>
#include <stdexcept>

#include "luaT.h"
#include "THTensor.h"
#include "THStorage.h"
//#include "clLuaHelper.h"
#include "LuaHelper.h"
#include "cudannWrapper.h"
#include "THCTensor.h"

using namespace std;

//#pragma message("compiling clnnWrapper")
THCState *getState(lua_State *L) {
    pushGlobal(L, "cutorch", "_state");
    void *state = lua_touserdata(L, -1);
//    cout << "state: " << (long)state << endl;
    lua_remove(L, -1);
    return (THCState *)state;
}
THCudaTensor *popCudaTensor(lua_State *L) {
    void **pTensor = (void **)lua_touserdata(L, -1);
    THCudaTensor *tensor = (THCudaTensor *)(*pTensor);
    lua_remove(L, -1);
    return tensor;
}
void pushCudaTensor(THCState *state, lua_State *L, THCudaTensor *tensor) {
    THCudaTensor_retain(state, tensor);
    luaT_pushudata(L, tensor, "torch.CudaTensor");    
}

