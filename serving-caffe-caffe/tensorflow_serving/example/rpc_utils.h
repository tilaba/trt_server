/* Copyright 2016 IBM Corp. All Rights Reserved. */

#ifndef TENSORFLOW_SERVING_EXAMPLE_RPC_UTILS_H_
#define TENSORFLOW_SERVING_EXAMPLE_RPC_UTILS_H_

#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc/grpc.h"

namespace rpc_util {
class RpcHandleBase {
 public:
  virtual ~RpcHandleBase() = default;
  virtual void Proceed() = 0;
};

// Class encompassing the state and logic needed to serve a
// unary encoding request asynchronously.
template <typename TRequest, typename TResponse>
class UnaryRequestHandleBase : public RpcHandleBase {
 public:
  UnaryRequestHandleBase(::grpc::ServerCompletionQueue* cq);
  virtual ~UnaryRequestHandleBase() = default;

  void Proceed() override;
  void Finish(::grpc::Status status);

  const TRequest& request() { return request_; }
  TResponse* mutable_response() { return &response_; }

 protected:
  virtual void Process() = 0;
  virtual void RequestAsyncUnary(TRequest* req) = 0;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ::grpc::ServerContext ctx_;
  // The producer-consumer queue where for asynchronous server notifications.
  ::grpc::ServerCompletionQueue* cq_;
  // The means to get back to the client.
  ::grpc::ServerAsyncResponseWriter<TResponse> responder_;

 private:
  // What we get from the client.
  TRequest request_;
  // What we send back to the client.
  TResponse response_;
  // The current serving state.
  enum { CREATE, PROCESS, FINISH } status_;
};

template <typename S, typename Req, typename Res>
struct UnaryRequestStub {
  typedef S Service;
  typedef Req Request;
  typedef Res Response;
  typedef UnaryRequestHandleBase<Req, Res> Handle;

  typedef void (S::*type)(::grpc::ServerContext*, Req*,
                          ::grpc::ServerAsyncResponseWriter<Res>*,
                          ::grpc::CompletionQueue*,
                          ::grpc::ServerCompletionQueue*, void*);

  constexpr UnaryRequestStub(type fn) : fn_{fn} {}
  constexpr type fn() { return fn_; }

  const type fn_;
};

template <typename TBatton, typename TStub>
class UnaryRequestFn final
    : public UnaryRequestHandleBase<typename TStub::Request,
                                    typename TStub::Response> {
 public:
  using TRequest = typename TStub::Request;
  using TResponse = typename TStub::Response;
  using TService = typename TStub::Service;
  using Fn = std::function<void(TBatton batton,
                                UnaryRequestHandleBase<TRequest, TResponse>*)>;

  static void Begin(TBatton batton, ::grpc::ServerCompletionQueue* cq,
                    TService* service, Fn fn, const TStub stub) {
    new UnaryRequestFn<TBatton, TStub>(batton, cq, service, fn, stub.fn());
  }

 protected:
  virtual void RequestAsyncUnary(TRequest* req) override {
    (service_->*serv_p_)(&(this->ctx_), req, &(this->responder_), this->cq_,
                         this->cq_, this);
  }

  virtual void Process() override {
    new UnaryRequestFn<TBatton, TStub>(batton_, this->cq_, service_, fn_,
                                       serv_p_);

    fn_(batton_, this);
  }

 private:
  UnaryRequestFn(TBatton batton, ::grpc::ServerCompletionQueue* cq,
                 TService* service, Fn fn, const typename TStub::type serv_p)
      : UnaryRequestHandleBase<TRequest, TResponse>(cq),
        serv_p_{serv_p},
        fn_{fn},
        service_{service},
        batton_{batton} {
    this->Proceed();
  }

  const typename TStub::type serv_p_;
  Fn fn_;
  TService* service_;
  TBatton batton_;
};

template <typename TBatton, typename TStub, typename Fn>
void begin_async_unary(TBatton&& batton, const TStub stub,
                       typename TStub::Service* s,
                       ::grpc::ServerCompletionQueue* cq, Fn&& call) {
  UnaryRequestFn<TBatton, TStub>::Begin(std::forward<TBatton>(batton), cq, s,
                                        std::forward<Fn>(call), stub);
};

/////////////////////////////////////////////
// Implementation
//

template <typename Req, typename Res>
UnaryRequestHandleBase<Req, Res>::UnaryRequestHandleBase(
    grpc::ServerCompletionQueue* cq)
    : cq_(cq), responder_(&ctx_), status_(CREATE) {}

template <typename Req, typename Res>
void UnaryRequestHandleBase<Req, Res>::Proceed() {
  if (status_ == CREATE) {
    // As part of the initial CREATE state, we *request* that the system
    // start processing Encode requests. In this request, "this" acts as
    // the tag uniquely identifying the request (so that different RequestHandle
    // instances can serve different requests concurrently), in this case
    // the memory address of this RequestHandle instance.
    RequestAsyncUnary(&request_);
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
  } else if (status_ == PROCESS) {
    Process();
  } else {
    GPR_ASSERT(status_ == FINISH);
    // deallocate this request
    delete this;
  }
}

template <typename Req, typename Res>
void UnaryRequestHandleBase<Req, Res>::Finish(grpc::Status status) {
  status_ = FINISH;
  responder_.Finish(response_, status, this);
}
}  // namespace rpc_util

#endif // TENSORFLOW_SERVING_EXAMPLE_RPC_UTILS_H_