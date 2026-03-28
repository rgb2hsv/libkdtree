/** \file
 * \brief Bidirectional iterators over \ref KDTree elements (in-order).
 * \ingroup libkdtree
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 */

#pragma once

#include <iterator>
#include <memory>
#include <type_traits>

#include "node.hpp"

namespace libkdtree
{

template <std::size_t K, typename Val, typename Acc, typename Dist, typename Cmp, typename Alloc>
class KDTree;

template <typename Val, typename Ref, typename Ptr>
class TreeIterator;

template <typename Val, typename Ref, typename Ptr>
bool operator==(TreeIterator<Val, Ref, Ptr> const&, TreeIterator<Val, Ref, Ptr> const&);

template <typename Val>
bool operator==(TreeIterator<Val, const Val&, const Val*> const&,
                TreeIterator<Val, Val&, Val*> const&);

template <typename Val>
bool operator==(TreeIterator<Val, Val&, Val*> const&,
                TreeIterator<Val, const Val&, const Val*> const&);

template <typename Val, typename Ref, typename Ptr>
bool operator!=(TreeIterator<Val, Ref, Ptr> const&, TreeIterator<Val, Ref, Ptr> const&);

template <typename Val>
bool operator!=(TreeIterator<Val, const Val&, const Val*> const&,
                TreeIterator<Val, Val&, Val*> const&);

template <typename Val>
bool operator!=(TreeIterator<Val, Val&, Val*> const&,
                TreeIterator<Val, const Val&, const Val*> const&);

//! \internal In-order traversal using parent links; used by \ref TreeIterator.
class BaseIterator
{
protected:
    using BaseConstPtr = NodeBase::BaseConstPtr;
    BaseConstPtr mNode;

    BaseIterator(BaseConstPtr pNode = nullptr) noexcept : mNode(pNode) {}
    BaseIterator(BaseIterator const& refThat) noexcept = default;
    BaseIterator& operator=(BaseIterator const&) noexcept = default;

    void Increment() noexcept
    {
        if (mNode->mRight) {
            mNode = mNode->mRight;
            while (mNode->mLeft) mNode = mNode->mLeft;
        } else {
            BaseConstPtr pParent = mNode->mParent;
            while (pParent && mNode == pParent->mRight) {
                mNode = pParent;
                pParent = mNode->mParent;
            }
            if (pParent)
                mNode = pParent;
        }
    }

    void Decrement() noexcept
    {
        if (!mNode->mParent) {
            mNode = mNode->mRight;
        } else if (mNode->mLeft) {
            BaseConstPtr pWalk = mNode->mLeft;
            while (pWalk->mRight) pWalk = pWalk->mRight;
            mNode = pWalk;
        } else {
            BaseConstPtr pParent = mNode->mParent;
            while (pParent && mNode == pParent->mLeft) {
                mNode = pParent;
                pParent = mNode->mParent;
            }
            if (pParent)
                mNode = pParent;
        }
    }

    template <std::size_t K, typename Val, typename Acc, typename Dist, typename Cmp, typename Alloc>
    friend class KDTree;
};

/**
 * \brief Bidirectional iterator referencing a \ref TreeNode "TreeNode"'s \c value.
 *
 * \tparam Val Stored element type.
 * \tparam Ref Reference type returned by `operator*` (e.g. `Val const&` for const trees).
 * \tparam Ptr Pointer type returned by `operator->`.
 */
template <typename Val, typename Ref, typename Ptr>
class TreeIterator : protected BaseIterator
{
public:
    using value_type = Val;
    using reference = Ref;
    using pointer = Ptr;
    using iterator = TreeIterator<Val, Val&, Val*>;
    using const_iterator = TreeIterator<Val, Val const&, Val const*>;
    using Self = TreeIterator<Val, Ref, Ptr>;
    using LinkConstType = TreeNode<Val> const*;
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;

    TreeIterator() = default;
    explicit TreeIterator(LinkConstType pNode) noexcept : BaseIterator(pNode) {}

    //! Convert mutable \ref iterator to \c const_iterator (omitted when \c Self is already \c iterator).
    template<typename R2, typename P2,
             typename = std::enable_if_t<
                 std::is_same_v<TreeIterator<Val, R2, P2>, iterator> &&
                 !std::is_same_v<TreeIterator<Val, Ref, Ptr>, iterator>>>
    TreeIterator(TreeIterator<Val, R2, P2> const& that) noexcept : BaseIterator(that)
    {
    }

    TreeIterator(TreeIterator const&) noexcept = default;
    TreeIterator& operator=(TreeIterator const&) noexcept = default;
    TreeIterator(TreeIterator&&) noexcept = default;
    TreeIterator& operator=(TreeIterator&&) noexcept = default;

    //! Underlying tree node pointer (e.g. for \c KDTree::erase).
    [[nodiscard]] LinkConstType node() const noexcept
    {
        return LinkConstType(mNode);
    }

    reference operator*() const { return LinkConstType(mNode)->mValue; }

    pointer operator->() const { return std::addressof(LinkConstType(mNode)->mValue); }

    Self& operator++() noexcept
    {
        Increment();
        return *this;
    }

    Self operator++(int) noexcept(std::is_nothrow_copy_constructible_v<Self>)
    {
        Self ret = *this;
        Increment();
        return ret;
    }

    Self& operator--() noexcept
    {
        Decrement();
        return *this;
    }

    Self operator--(int) noexcept(std::is_nothrow_copy_constructible_v<Self>)
    {
        Self ret = *this;
        Decrement();
        return ret;
    }

    friend bool operator== <>(TreeIterator<Val, Ref, Ptr> const&, TreeIterator<Val, Ref, Ptr> const&);

    friend bool operator== <>(TreeIterator<Val, const Val&, const Val*> const&,
                               TreeIterator<Val, Val&, Val*> const&);

    friend bool operator== <>(TreeIterator<Val, Val&, Val*> const&,
                               TreeIterator<Val, const Val&, const Val*> const&);

    friend bool operator!= <>(TreeIterator<Val, Ref, Ptr> const&, TreeIterator<Val, Ref, Ptr> const&);

    friend bool operator!= <>(TreeIterator<Val, const Val&, const Val*> const&,
                               TreeIterator<Val, Val&, Val*> const&);

    friend bool operator!= <>(TreeIterator<Val, Val&, Val*> const&,
                               TreeIterator<Val, const Val&, const Val*> const&);
};

template <typename Val, typename Ref, typename Ptr>
bool operator==(TreeIterator<Val, Ref, Ptr> const& refX, TreeIterator<Val, Ref, Ptr> const& refY)
{
    return refX.mNode == refY.mNode;
}

template <typename Val>
bool operator==(TreeIterator<Val, const Val&, const Val*> const& refX,
                TreeIterator<Val, Val&, Val*> const& refY)
{
    return refX.mNode == refY.mNode;
}

template <typename Val>
bool operator==(TreeIterator<Val, Val&, Val*> const& refX,
                TreeIterator<Val, const Val&, const Val*> const& refY)
{
    return refX.mNode == refY.mNode;
}

template <typename Val, typename Ref, typename Ptr>
bool operator!=(TreeIterator<Val, Ref, Ptr> const& refX, TreeIterator<Val, Ref, Ptr> const& refY)
{
    return refX.mNode != refY.mNode;
}

template <typename Val>
bool operator!=(TreeIterator<Val, const Val&, const Val*> const& refX,
                TreeIterator<Val, Val&, Val*> const& refY)
{
    return refX.mNode != refY.mNode;
}

template <typename Val>
bool operator!=(TreeIterator<Val, Val&, Val*> const& refX,
                TreeIterator<Val, const Val&, const Val*> const& refY)
{
    return refX.mNode != refY.mNode;
}

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
