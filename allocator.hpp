/** \file
 * \brief Node allocation helpers for \ref KDTree (RAII guard for exception safety).
 * \ingroup libkdtree
 * \author Martin F. Krafft <libkdtree@pobox.madduck.net>
 */

#pragma once

#include <cstddef>

#include "node.hpp"

namespace libkdtree
{

//! CRTP-style base: allocate / construct / destroy \ref TreeNode "TreeNode<Tp>" with \p Alloc.
template <typename Tp, typename Alloc>
class AllocBase
{
public:
    using NodeType = TreeNode<Tp>;
    using BasePtr = typename NodeType::BasePtr;
    using allocator_type = Alloc;

    explicit AllocBase(allocator_type const& refAlloc) : mNodeAlloc(refAlloc) {}

    [[nodiscard]] allocator_type getAllocator() const { return mNodeAlloc; }

    /**
     * \brief Allocates one node; deallocates in the destructor unless \ref disconnect().
     *
     * Typical use: `NoLeakAlloc guard(this); constructNode(guard.get(), ...); guard.disconnect();`
     */
    class NoLeakAlloc
    {
        AllocBase* mBase;
        NodeType* mNewNode;

    public:
        explicit NoLeakAlloc(AllocBase* base_in)
            : mBase(base_in), mNewNode(mBase->allocateNode())
        {
        }

        NoLeakAlloc(NoLeakAlloc const&) = delete;
        NoLeakAlloc& operator=(NoLeakAlloc const&) = delete;
        NoLeakAlloc(NoLeakAlloc&&) = delete;
        NoLeakAlloc& operator=(NoLeakAlloc&&) = delete;

        //! Pointer to uninitialized storage (placement-new in \ref constructNode).
        NodeType* get() { return mNewNode; }
        //! Transfers ownership to caller; destructor becomes a no-op.
        void disconnect() { mNewNode = nullptr; }

        ~NoLeakAlloc()
        {
            if (mNewNode) mBase->deallocateNode(mNewNode);
        }
    };

protected:
    allocator_type mNodeAlloc;

    NodeType* allocateNode() { return mNodeAlloc.allocate(1); }

    void deallocateNode(NodeType* const pNode) { mNodeAlloc.deallocate(pNode, 1); }

    void constructNode(NodeType* pNode, Tp const& refVal = Tp(), BasePtr const pParent = nullptr,
                       BasePtr const pLeft = nullptr, BasePtr const pRight = nullptr)
    {
        new (pNode) NodeType(refVal, pParent, pLeft, pRight);
    }

    void destroyNode(NodeType* pNode) { pNode->~NodeType(); }
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
