/** \file
 * \brief Default accessor and distance functors for \ref KDTree.
 * \ingroup libkdtree
 *
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 * \author Sylvain Bougerel <sylvain.bougerel.devel@gmail.com>
 */

#pragma once

#include <cstddef>

namespace libkdtree
{

//! Default \c Acc: reads `refVal[uIdx]`; requires `Val::value_type` and index operator.
template <typename Val>
struct BracketAccessor {
    using result_type = typename Val::value_type;

    constexpr result_type operator()(Val const& refVal, std::size_t uIdx) const
    {
        return refVal[uIdx];
    }
};

//! Predicate that accepts every \c Tp; used as default for unfiltered nearest-neighbor search.
template <typename Tp>
struct AlwaysTrue {
    bool operator()(const Tp&) const { return true; }
};

//! Per-axis squared difference `(refA - refB)^2`; summed over dimensions for Euclidean-style distance.
template <typename Tp, typename Dist>
struct SquaredDifference {
    using distance_type = Dist;

    constexpr distance_type operator()(Tp const& refA, Tp const& refB) const
    {
        distance_type const distElem = refA - refB;
        return distElem * distElem;
    }
};

//! Like \ref SquaredDifference but increments a mutable call counter (profiling / tests).
template <typename Tp, typename Dist>
struct SquaredDifferenceCounted {
    using distance_type = Dist;

    SquaredDifferenceCounted() : m_cCalls(0) {}

    void reset() { m_cCalls = 0; }

    //! Reference to the number of invocations of `operator()`.
    long& count() const { return m_cCalls; }

    distance_type operator()(const Tp& refA, const Tp& refB) const
    {
        distance_type distElem = refA - refB;
        ++m_cCalls;
        return distElem * distElem;
    }

private:
    mutable long m_cCalls;
};

}  // namespace libkdtree

/* COPYRIGHT --
 *
 * This file is part of libkdtree++, a C++ template KD-Tree sorting container.
 * libkdtree++ is (c) 2004-2007 Martin F. Krafft <libkdtree@pobox.madduck.net>
 * and Sylvain Bougerel <sylvain.bougerel.devel@gmail.com> distributed under the
 * terms of the Artistic License 2.0. See the ./COPYING file in the source tree
 * root for more information.
 *
 * THIS PACKAGE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
 * OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
