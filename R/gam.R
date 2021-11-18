library(mgcv)
import::from(pracma, linspace)
import::from(purrr, map)


wrap_s = function(s) {
  return(paste(c('s(', s, ')'), collapse=''))
}

file_append = function(s, stdout_file) {
    cat(s, file=stdout_file, sep='\n', append=TRUE)
}


fit_gam = function(label, df_data, stdout_file, image_dir) {
    set.seed(1)

    if (substr(image_dir, nchar(image_dir), nchar(image_dir)) != "/") {
        image_dir = paste0(image_dir, '/')
    }

    print("stdout file:")
    print(stdout_file)

    print("image_dir:")
    print(image_dir)

    # Fit GAM
    indep = names(df_data)[!(names(df_data) %in% c("response"))]
    gam_formula = as.formula(paste("response ~", paste(map(indep, wrap_s), collapse="+")))

    fitted_gam = gam(gam_formula, method="REML", family=quasibinomial(link="logit"), data=df_data)
    file_append(paste("Index:", label), stdout_file)
    file_append(capture.output(print(summary(fitted_gam))), stdout_file)

    # Plot and save file.
    png(
      paste0(image_dir, paste0('gam_curves', label, '.png')),
      width=1500, height=1000, pointsize=15
    )
    plot(fitted_gam, residuals=FALSE, rug=TRUE, se=FALSE, shade=FALSE, pages=1, all.terms=FALSE)
    dev.off()

    # Return fitted GAM.
    return(fitted_gam)
}
