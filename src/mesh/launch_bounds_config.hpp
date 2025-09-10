#pragma once

#ifndef NOVAPP_LB_TB
#  define NOVAPP_LB_TB 256
#endif
#ifndef NOVAPP_LB_MINBLK
#  define NOVAPP_LB_MINBLK 2
#endif

namespace novapp {
  static constexpr unsigned int maxTperB  = NOVAPP_LB_TB;
  static constexpr unsigned int minBperSM = NOVAPP_LB_MINBLK;
}
