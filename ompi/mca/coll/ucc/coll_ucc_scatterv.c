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
static inline ucc_status_t
mca_coll_ucc_scatterv_init_common(const void *sbuf, ompi_count_array_t scounts,
                                  ompi_disp_array_t disps, struct ompi_datatype_t *sdtype,
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
    int max_scounts;
    int i;

    /* Local checks - convert datatypes if valid */
    if (comm_rank == root) {
        max_scounts = scounts[0];
        for (i = 1; i < comm_size; ++i) {
            if (max_scounts < scounts[i]) {
                max_scounts = scounts[i];
            }
        }
        if ((is_inplace || ompi_datatype_is_contiguous_memory_layout(rdtype, rcount)) &&
            ompi_datatype_is_contiguous_memory_layout(sdtype, max_scounts)) {
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

    flags = (ompi_count_array_is_64bit(scounts) ? UCC_COLL_ARGS_FLAG_COUNT_64BIT : 0) |
            (ompi_disp_array_is_64bit(disps) ? UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT : 0) |
            (is_inplace ? UCC_COLL_ARGS_FLAG_IN_PLACE : 0) |
            (persistent ? UCC_COLL_ARGS_FLAG_PERSISTENT : 0);
    /* Build coll args - will have COLL_UCC_DT_UNSUPPORTED if local checks failed */
    ucc_coll_args_t coll = {
        .mask      = flags ? UCC_COLL_ARGS_FIELD_FLAGS : 0,
        .flags     = flags,
        .coll_type = UCC_COLL_TYPE_SCATTERV,
        .root      = root,
        .src.info_v = {
            .buffer        = (void*)sbuf,
            .counts        = (ucc_count_t*)ompi_count_array_ptr(scounts),
            .displacements = (ucc_aint_t*)ompi_disp_array_ptr(disps),
            .datatype      = ucc_sdt,
            .mem_type      = UCC_MEMORY_TYPE_UNKNOWN
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
            UCC_VERBOSE(5, "scatterv: non-contiguous datatype detected across ranks");
        } else if (status == UCC_ERR_INVALID_PARAM) {
            UCC_VERBOSE(5, "scatterv: asymmetric datatypes or memory types detected");
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
mca_coll_ucc_scatterv_init_common(const void *sbuf, ompi_count_array_t scounts,
                                  ompi_disp_array_t disps, struct ompi_datatype_t *sdtype,
                                  void *rbuf, size_t rcount,
                                  struct ompi_datatype_t *rdtype, int root,
                                  bool persistent, mca_coll_ucc_module_t *ucc_module,
                                  ucc_coll_req_h *req,
                                  mca_coll_ucc_req_t *coll_req)
{
    ucc_datatype_t ucc_sdt = UCC_DT_INT8, ucc_rdt = UCC_DT_INT8;
    bool is_inplace = (MPI_IN_PLACE == rbuf);
    int comm_rank = ompi_comm_rank(ucc_module->comm);
    uint64_t flags = 0;
    if (comm_rank == root) {
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
        ucc_rdt = ompi_dtype_to_ucc_dtype(rdtype);
        if (COLL_UCC_DT_UNSUPPORTED == ucc_rdt) {
            UCC_VERBOSE(5, "ompi_datatype is not supported: dtype = %s",
                        rdtype->super.name);
            goto fallback;
        }
    }

    flags = (ompi_count_array_is_64bit(scounts) ? UCC_COLL_ARGS_FLAG_COUNT_64BIT : 0) |
            (ompi_disp_array_is_64bit(disps) ? UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT : 0) |
            (is_inplace ? UCC_COLL_ARGS_FLAG_IN_PLACE : 0) |
            (persistent ? UCC_COLL_ARGS_FLAG_PERSISTENT : 0);

    ucc_coll_args_t coll = {
        .mask      = flags ? UCC_COLL_ARGS_FIELD_FLAGS : 0,
        .flags     = flags,
        .coll_type = UCC_COLL_TYPE_SCATTERV,
        .root      = root,
        .src.info_v = {
            .buffer        = (void*)sbuf,
            .counts        = (ucc_count_t*)ompi_count_array_ptr(scounts),
            .displacements = (ucc_aint_t*)ompi_disp_array_ptr(disps),
            .datatype      = ucc_sdt,
            .mem_type      = UCC_MEMORY_TYPE_UNKNOWN
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

int mca_coll_ucc_scatterv(const void *sbuf, ompi_count_array_t scounts,
                          ompi_disp_array_t disps, struct ompi_datatype_t *sdtype,
                          void *rbuf, size_t rcount,
                          struct ompi_datatype_t *rdtype, int root,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module)
{
    mca_coll_ucc_module_t *ucc_module = (mca_coll_ucc_module_t*)module;
    ucc_coll_req_h         req;

    UCC_VERBOSE(3, "running ucc scatterv");
    COLL_UCC_CHECK(mca_coll_ucc_scatterv_init_common(sbuf, scounts, disps, sdtype,
                                                     rbuf, rcount, rdtype, root,
                                                     false, ucc_module, &req, NULL));
    COLL_UCC_POST_AND_CHECK(req);
    COLL_UCC_CHECK(coll_ucc_req_wait(req));
    return OMPI_SUCCESS;
fallback:
    UCC_VERBOSE(3, "running fallback scatterv");
    return ucc_module->previous_scatterv(sbuf, scounts, disps, sdtype, rbuf,
                                         rcount, rdtype, root, comm,
                                         ucc_module->previous_scatterv_module);
}

int mca_coll_ucc_iscatterv(const void *sbuf, ompi_count_array_t scounts,
                           ompi_disp_array_t disps, struct ompi_datatype_t *sdtype,
                           void *rbuf, size_t rcount,
                           struct ompi_datatype_t *rdtype, int root,
                           struct ompi_communicator_t *comm,
                           ompi_request_t** request,
                           mca_coll_base_module_t *module)
{
    mca_coll_ucc_module_t *ucc_module = (mca_coll_ucc_module_t*)module;
    ucc_coll_req_h         req;
    mca_coll_ucc_req_t    *coll_req = NULL;

    UCC_VERBOSE(3, "running ucc iscatterv");
    COLL_UCC_GET_REQ(coll_req, comm);
    COLL_UCC_CHECK(mca_coll_ucc_scatterv_init_common(sbuf, scounts, disps, sdtype,
                                                     rbuf, rcount, rdtype, root,
                                                     false, ucc_module, &req, coll_req));
    COLL_UCC_POST_AND_CHECK(req);
    *request = &coll_req->super;
    return OMPI_SUCCESS;
fallback:
    UCC_VERBOSE(3, "running fallback iscatterv");
    if (coll_req) {
        mca_coll_ucc_req_free((ompi_request_t **)&coll_req);
    }
    return ucc_module->previous_iscatterv(sbuf, scounts, disps, sdtype, rbuf,
                                          rcount, rdtype, root, comm, request,
                                          ucc_module->previous_iscatterv_module);
}

int mca_coll_ucc_scatterv_init(const void *sbuf, ompi_count_array_t scounts,
                               ompi_disp_array_t disps, struct ompi_datatype_t *sdtype, void *rbuf,
                               size_t rcount, struct ompi_datatype_t *rdtype, int root,
                               struct ompi_communicator_t *comm, struct ompi_info_t *info,
                               ompi_request_t **request, mca_coll_base_module_t *module)
{
    mca_coll_ucc_module_t *ucc_module = (mca_coll_ucc_module_t *) module;
    ucc_coll_req_h req;
    mca_coll_ucc_req_t *coll_req = NULL;

    COLL_UCC_GET_REQ_PERSISTENT(coll_req, comm);
    UCC_VERBOSE(3, "scatterv_init init %p", coll_req);
    COLL_UCC_CHECK(mca_coll_ucc_scatterv_init_common(sbuf, scounts, disps, sdtype,
                                                     rbuf, rcount, rdtype, root,
                                                     true, ucc_module, &req, coll_req));
    *request = &coll_req->super;
    return OMPI_SUCCESS;
fallback:
    UCC_VERBOSE(3, "running fallback scatterv_init");
    if (coll_req) {
        mca_coll_ucc_req_free((ompi_request_t **) &coll_req);
    }
    return ucc_module->previous_scatterv_init(sbuf, scounts, disps, sdtype, rbuf, rcount, rdtype,
                                              root, comm, info, request,
                                              ucc_module->previous_scatterv_init_module);
}
