add_mlir_library(
  ZuanInterfaces

  ZuanInferShapeInterface.cpp
  ZuanUnrollingInterface.cpp

  DEPENDS
  MLIRZuanInferShapeInterfaceIncGen
  MLIRZuanUnrollingInterfaceIncGen
)
