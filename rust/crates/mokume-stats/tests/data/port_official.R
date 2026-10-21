# Regenerate the independent PORT regression trajectories; see JSON source hashes.
library(jsonlite)
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop('Usage: Rscript port_official.R output.json')
if (getRversion() != '4.5.3') stop('The pinned reference is R 4.5.3')
cases <- list(
 list(name='coupled_quadratic', start=c(4,-3),lower=c(-Inf,-Inf),
      f=function(x) (x[1]-1)^2+2*(x[2]+2)^2+0.4*(x[1]-1)*(x[2]+2),
      g=function(x)c(2*(x[1]-1)+.4*(x[2]+2),4*(x[2]+2)+.4*(x[1]-1)),
      h=function(x)matrix(c(2,.4,.4,4),2)),
 list(name='active_lower_bound', start=c(0,0),lower=c(0,0),
      f=function(x)(x[1]+2)^2+3*(x[2]-1)^2,
      g=function(x)c(2*(x[1]+2),6*(x[2]-1)),
      h=function(x)diag(c(2,6))),
 list(name='rosenbrock', start=c(-1.2,1),lower=c(-2,-Inf),
      f=function(x)100*(x[2]-x[1]^2)^2+(1-x[1])^2,
      g=function(x)c(-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]),200*(x[2]-x[1]^2)),
      h=function(x)matrix(c(1200*x[1]^2-400*x[2]+2,-400*x[1],-400*x[1],200),2)),
 list(name='different_scales', start=c(0,0),lower=c(0,0),
      f=function(x).001*(x[1]-30)^2+100*(x[2]-.2)^2,
      g=function(x)c(.002*(x[1]-30),200*(x[2]-.2)),
      h=function(x)diag(c(.002,200)))
)
cases <- c(cases, list(
 list(name='indefinite_quartic',start=c(0,.2),lower=c(-Inf,0),
  f=function(x)(x[1]^2-1)^2+(x[2]-.4)^2,
  g=function(x)c(4*x[1]*(x[1]^2-1),2*(x[2]-.4)),
  h=function(x)diag(c(12*x[1]^2-4,2))),
 list(name='coupled_lower_bound',start=c(0,2),lower=c(0,0),
  f=function(x)(x[1]-x[2])^2+10*(x[2]+1)^2,
  g=function(x)c(2*(x[1]-x[2]),2*(x[2]-x[1])+20*(x[2]+1)),
  h=function(x)matrix(c(2,-2,-2,22),2))
))
results <- lapply(cases,function(c){
  evaluations<-list()
  fn<-function(x){val<-c$f(x);evaluations[[length(evaluations)+1L]]<<-list(x=x,f=val);val}
  out<-nlminb(c$start,fn,c$g,c$h,lower=c$lower)
  out$status <- as.integer(sub('.*\\(([0-9]+)\\)$', '\\1', out$message))
  c$lower[!is.finite(c$lower)] <- NA_real_
  list(name=c$name,start=c$start,lower=c$lower,expected=out,evaluations=evaluations)
})
write_json(list(upstream=R.version.string,entrypoint='stats::nlminb(analytic gradient, analytic Hessian, default control)',cases=results),args[[1]],digits=NA,auto_unbox=TRUE,pretty=TRUE,na="null")
