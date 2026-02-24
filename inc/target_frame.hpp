#ifndef TARGET_FRAME_HPP
#define TARGET_FRAME_HPP

// =================================================================================================
// Shared TARGET_FRAME constant declaration for saving debug outputs
// =================================================================================================
// This constant controls which frame's data is saved to binary files for comparison with Python.
// Set to the frame number you want to save (0-indexed, e.g., 69 for frame 69).
// Set to -1 to disable saving.
// 
// The constant is defined in target_frame.cpp and declared here as extern.
// 
// This constant is used in:
//   - dpvo.cpp: Save BA input parameters
//   - ba.cpp: Save BA intermediate outputs
//   - patchify.cpp: Save ONNX model outputs (FNet/INet) and patchify outputs (coords, gmap, imap, patches)
//   - update_onnx.cpp: Save update model inputs/outputs
// =================================================================================================

extern const int TARGET_FRAME;  // Defined in target_frame.cpp

#endif // TARGET_FRAME_HPP

