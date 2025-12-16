/**
 * Copyright (c) 2021 Mellanox Technologies. All rights reserved.
 * Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
 * Copyright (c) 2025      Fujitsu Limited. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 */

#include "coll_ucc_common.h"

#if UCC_API_VERSION >= UCC_VERSION(1, 7)
static inline
mca_coll_ucc_scatter_init_common(const void *sbuf, size_t scount,
                                 struct ompi_datatype_t *sdtype,
                                 void *rbuf, size_t rcount,
                                 struct ompi_datatype_t *rdtype, int root,
                                 bool persistent, mca_coll_ucc_module_t *ucc_module,
                                 ucc_coll_req_h *req,
                                 mca_coll_ucc_req_t *coll_req)
{
    ucc_datatype_t ucc_sdt = COLL_UCC_DT_UNSUPPORTED, ucc_rdt = COLL_UCC_DT_UNSUPPORTED;
    bool is_inplace = (MPI_IN_PLACE == rbuf);
    int comm_rank = ompi_comm_rank(ucc_module->comm);
    int comm_size = ompi_comm_size(ucc_module->comm);
    uint64_t flags = 0;
    ucc_status_t status;

    /* Local checks - convert datatypes if valid */
    if (comm_rank == root) {
        if ((is_inplace || ompi_datatype_is_contiguous_memory_layout(rdtype, rcount)) &&
            ompi_datatype_is_contiguous_memory_layout(sdtype, scount * comm_size)) {
            ucc_sdt = ompi_dtype_to_ucc_dtype(sdtype);
            if (!is_inplace) {
                ucc_rdt = ompi_dtype_to_ucc_dtype(rdtype);
            }

            if ((COLL_UCC_DT_UNSUPPORTED == ucc_sdt) ||
                (COLL_UCC_DT_UNSUPPORTED == ucc_rdt)) {
                UCC_VERBOSE(5, "ompi_datatype is not supported: dtype = %s",
                            (COLL_UCC_DT_UNSUPPORTED == ucc_sdt) ?
                            sdtype->super.name : rdtype->super.name);
            }
        }
    } else {
        if (ompi_datatype_is_contiguous_memory_layout(rdtype, rcount)) {
            ucc_rdt = ompi_dtype_to_ucc_dtype(rdtype);
            if (COLL_UCC_DT_UNSUPPORTED == ucc_rdt) {
                UCC_VERBOSE(5, "ompi_datatype is not supported: dtype = %s",
                            rdtype->super.name);
            }
        } else {
            ucc_rdt = COLL_UCC_DT_UNSUPPORTED;
        }
    }

    /* Build coll args - will have COLL_UCC_DT_UNSUPPORTED if local checks failed */
    flags = (is_inplace ? UCC_COLL_ARGS_FLAG_IN_PLACE : 0) |
            (persistent ? UCC_COLL_ARGS_FLAG_PERSISTENT : 0);

    ucc_coll_args_t coll = {
        .mask      = flags ? UCC_COLL_ARGS_FIELD_FLAGS : 0,
        .flags     = flags,
        .coll_type = UCC_COLL_TYPE_SCATTER,
        .root      = root,
        .src.info  = {
            .buffer   = (void*)sbuf,
            .count    = scount * comm_size,
            .datatype = ucc_sdt,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
        .dst.info = {
            .buffer   = (void*)rbuf,
            .count    = rcount,
            .datatype = ucc_rdt,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
    };

    /* COLLECTIVE CHECK - ALL ranks participate, detects all issues */
    status = ucc_dt_check_rooted_collective(ucc_module->ucc_team, &coll, comm_rank);
    if (UCC_OK != status) {
        if (status == UCC_ERR_NOT_SUPPORTED) {
            UCC_VERBOSE(5, "scatter: non-contiguous datatype detected across ranks");
        } else if (status == UCC_ERR_INVALID_PARAM) {
            UCC_VERBOSE(5, "scatter: asymmetric datatypes or memory types detected");
        }
        goto fallback;
    }

    COLL_UCC_REQ_INIT(coll_req, req, coll, ucc_module);
    return UCC_OK;
fallback:
    return UCC_ERR_NOT_SUPPORTED;
}
#else /* UCC_API_VERSION < UCC_VERSION(1, 7) */
static inline ucc_status_t
mca_coll_ucc_scatter_init_common(const void *sbuf, size_t scount,
                                 struct ompi_datatype_t *sdtype,
                                 void *rbuf, size_t rcount,
                                 struct ompi_datatype_t *rdtype, int root,
                                 bool persistent, mca_coll_ucc_module_t *ucc_module,
                                 ucc_coll_req_h *req,
                                 mca_coll_ucc_req_t *coll_req)
{
    ucc_datatype_t ucc_sdt = UCC_DT_INT8, ucc_rdt = UCC_DT_INT8;
    bool is_inplace = (MPI_IN_PLACE == rbuf);
    int comm_rank = ompi_comm_rank(ucc_module->comm);
    int comm_size = ompi_comm_size(ucc_module->comm);
    uint64_t flags = 0;

    if (comm_rank == root) {
        if (!(is_inplace || ompi_datatype_is_contiguous_memory_layout(rdtype, rcount)) ||
            !ompi_datatype_is_contiguous_memory_layout(sdtype, scount * comm_size)) {
            goto fallback;
        }

        ucc_sdt = ompi_dtype_to_ucc_dtype(sdtype);
        if (!is_inplace) {
            ucc_rdt = ompi_dtype_to_ucc_dtype(rdtype);
        }

        if ((COLL_UCC_DT_UNSUPPORTED == ucc_sdt) ||
            (COLL_UCC_DT_UNSUPPORTED == ucc_rdt)) {
            UCC_VERBOSE(5, "ompi_datatype is not supported: dtype = %s",
                        (COLL_UCC_DT_UNSUPPORTED == ucc_sdt) ?
                        sdtype->super.name : rdtype->super.name);
            goto fallback;
        }
    } else {
        if (!ompi_datatype_is_contiguous_memory_layout(rdtype, rcount)) {
            goto fallback;
        }

        ucc_rdt = ompi_dtype_to_ucc_dtype(rdtype);
        if (COLL_UCC_DT_UNSUPPORTED == ucc_rdt) {
            UCC_VERBOSE(5, "ompi_datatype is not supported: dtype = %s",
                        rdtype->super.name);
            goto fallback;
        }
    }

    flags = (is_inplace ? UCC_COLL_ARGS_FLAG_IN_PLACE : 0) |
            (persistent ? UCC_COLL_ARGS_FLAG_PERSISTENT : 0);

    ucc_coll_args_t coll = {
        .mask      = flags ? UCC_COLL_ARGS_FIELD_FLAGS : 0,
        .flags     = flags,
        .coll_type = UCC_COLL_TYPE_SCATTER,
        .root      = root,
        .src.info  = {
            .buffer   = (void*)sbuf,
            .count    = scount * comm_size,
            .datatype = ucc_sdt,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
        .dst.info = {
            .buffer   = (void*)rbuf,
            .count    = rcount,
            .datatype = ucc_rdt,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN
        },
    };

    COLL_UCC_REQ_INIT(coll_req, req, coll, ucc_module);
    return UCC_OK;
fallback:
    return UCC_ERR_NOT_SUPPORTED;
}
#endif /* UCC_API_VERSION */

int mca_coll_ucc_scatter(const void *sbuf, size_t scount,
                         struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                         struct ompi_datatype_t *rdtype, int root,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
    mca_coll_ucc_module_t *ucc_module = (mca_coll_ucc_module_t*)module;
    ucc_coll_req_h         req;

    UCC_VERBOSE(3, "running ucc scatter");
    COLL_UCC_CHECK(mca_coll_ucc_scatter_init_common(sbuf, scount, sdtype, rbuf, rcount,
                                                    rdtype, root, false, ucc_module, &req,
                                                    NULL));
    COLL_UCC_POST_AND_CHECK(req);
    COLL_UCC_CHECK(coll_ucc_req_wait(req));
    return OMPI_SUCCESS;
fallback:
    UCC_VERBOSE(3, "running fallback scatter");
    return ucc_module->previous_scatter(sbuf, scount, sdtype, rbuf, rcount,
                                        rdtype, root, comm,
                                        ucc_module->previous_scatter_module);

}

int mca_coll_ucc_iscatter(const void *sbuf, size_t scount,
                         struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                         struct ompi_datatype_t *rdtype, int root,
                         struct ompi_communicator_t *comm,
                         ompi_request_t** request,
                         mca_coll_base_module_t *module)
{
    mca_coll_ucc_module_t *ucc_module = (mca_coll_ucc_module_t*)module;
    ucc_coll_req_h         req;
    mca_coll_ucc_req_t    *coll_req = NULL;

    UCC_VERBOSE(3, "running ucc iscatter");
    COLL_UCC_GET_REQ(coll_req, comm);
    COLL_UCC_CHECK(mca_coll_ucc_scatter_init_common(sbuf, scount, sdtype, rbuf, rcount,
                                                    rdtype, root, false, ucc_module, &req,
                                                    coll_req));
    COLL_UCC_POST_AND_CHECK(req);
    *request = &coll_req->super;
    return OMPI_SUCCESS;
fallback:
    UCC_VERBOSE(3, "running fallback iscatter");
    if (coll_req) {
        mca_coll_ucc_req_free((ompi_request_t **)&coll_req);
    }
    return ucc_module->previous_iscatter(sbuf, scount, sdtype, rbuf, rcount,
                                         rdtype, root, comm, request,
                                         ucc_module->previous_iscatter_module);
}

int mca_coll_ucc_scatter_init(const void *sbuf, size_t scount, struct ompi_datatype_t *sdtype,
                              void *rbuf, size_t rcount, struct ompi_datatype_t *rdtype, int root,
                              struct ompi_communicator_t *comm, struct ompi_info_t *info,
                              ompi_request_t **request, mca_coll_base_module_t *module)
{
    mca_coll_ucc_module_t *ucc_module = (mca_coll_ucc_module_t *) module;
    ucc_coll_req_h req;
    mca_coll_ucc_req_t *coll_req = NULL;

    COLL_UCC_GET_REQ_PERSISTENT(coll_req, comm);
    UCC_VERBOSE(3, "scatter_init init %p", coll_req);
    COLL_UCC_CHECK(mca_coll_ucc_scatter_init_common(sbuf, scount, sdtype, rbuf, rcount,
                                                    rdtype, root, true, ucc_module, &req,
                                                    coll_req));
    *request = &coll_req->super;
    return OMPI_SUCCESS;
fallback:
    UCC_VERBOSE(3, "running fallback scatter_init");
    if (coll_req) {
        mca_coll_ucc_req_free((ompi_request_t **) &coll_req);
    }
    return ucc_module->previous_scatter_init(sbuf, scount, sdtype, rbuf, rcount, rdtype, root, comm,
                                             info, request,
                                             ucc_module->previous_scatter_init_module);
}
