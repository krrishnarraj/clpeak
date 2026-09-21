# The LiteRT AAR's Java classes reference its `litert-api` companion, which
# is excluded from the build (only the AAR's native libraries are wanted;
# clpeak talks to LiteRT's C API through the FFI library).  R8 must not treat
# the dangling references as an error: nothing ever loads those classes.
-dontwarn org.tensorflow.lite.**
-dontwarn com.google.ai.edge.litert.**
