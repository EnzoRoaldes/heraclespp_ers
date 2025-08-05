#pragma once

#include <Kokkos_Core.hpp>
#include <string>

// 3D loop utility for Idefix-style reconstruction
template <typename Function>
inline void idefix_for(const std::string & NAME,
                       const int & KB, const int & KE,
                       const int & JB, const int & JE,
                       const int & IB, const int & IE,
                       Function function) {
    const int NK = KE - KB;
    const int NJ = JE - JB;
    const int NI = IE - IB;
    const int NKNJNI = NK * NJ * NI;
    const int NJNI = NJ * NI;

    Kokkos::parallel_for(NAME, NKNJNI,
        KOKKOS_LAMBDA (const int& IDX) {
            int k = IDX / NJNI;
            int j = (IDX - k * NJNI) / NI;
            int i = IDX - k * NJNI - j * NI;
            k += KB;
            j += JB;
            i += IB;
            function(i, j, k);
        }
    );
}